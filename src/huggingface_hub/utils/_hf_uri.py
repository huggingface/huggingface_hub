"""Central parsing for ``hf://`` URIs and plain HF identifiers."""

import re
from dataclasses import dataclass
from typing import Optional, Union
from urllib.parse import unquote

from .. import constants


# ---------------------------------------------------------------------------
# Regex for special revision references that contain "/" characters
# (e.g. refs/convert/parquet, refs/pr/123).  Moved here from
# hf_file_system.py so every consumer can share it.
# ---------------------------------------------------------------------------
SPECIAL_REFS_REVISION_REGEX = re.compile(
    r"""
    (^refs\/convert\/\w+)     # `refs/convert/parquet` revisions
    |
    (^refs\/pr\/\d+)          # PR revisions
    """,
    re.VERBOSE,
)

# Mapping that includes "buckets" alongside the standard repo-type plurals.
_HF_TYPES_MAPPING: dict[str, str] = {
    **constants.REPO_TYPES_MAPPING,  # datasets, spaces, models
    "buckets": "bucket",
}

HF_PREFIX = "hf://"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedHfUrl:
    """Result of parsing an ``hf://`` URI or plain identifier for a repository."""

    repo_type: str
    """Canonical repo type: ``"model"``, ``"dataset"``, or ``"space"``."""

    repo_id: str
    """Repository identifier, e.g. ``"namespace/name"`` or ``"name"``."""

    revision: Optional[str]
    """Revision parsed from the ``@revision`` syntax, or *None* if absent."""

    path_in_repo: str
    """Sub-path inside the repository (empty string when absent)."""

    has_explicit_type: bool = False
    """Whether a type prefix (``datasets/``, ``spaces/``, ``models/``) was present in the input."""

    @property
    def namespace(self) -> Optional[str]:
        """The namespace (user/org) portion of the repo_id, or *None* for canonical repos."""
        if "/" in self.repo_id:
            return self.repo_id.split("/", 1)[0]
        return None

    @property
    def repo_name(self) -> str:
        """The repo name portion of the repo_id (without namespace)."""
        if "/" in self.repo_id:
            return self.repo_id.split("/", 1)[1]
        return self.repo_id


@dataclass(frozen=True)
class ParsedBucketUrl:
    """Result of parsing an ``hf://buckets/`` URI or plain bucket identifier."""

    bucket_id: str
    """Bucket identifier: ``"namespace/name"``."""

    path: str
    """Prefix / sub-path inside the bucket (empty string when absent)."""

    @property
    def namespace(self) -> Optional[str]:
        """The namespace (user/org) portion of the bucket_id, or *None*."""
        if "/" in self.bucket_id:
            return self.bucket_id.split("/", 1)[0]
        return None

    @property
    def bucket_name(self) -> str:
        """The bucket name portion of the bucket_id (without namespace)."""
        if "/" in self.bucket_id:
            return self.bucket_id.split("/", 1)[1]
        return self.bucket_id


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------
def parse_hf_url(
    url: str,
    *,
    default_type: str = constants.REPO_TYPE_MODEL,
) -> Union[ParsedHfUrl, ParsedBucketUrl]:
    """Parse an ``hf://`` URI or plain HF identifier into structured components.

    Pure parsing only - **no HTTP calls, no existence validation**.

    Accepted formats::

        hf://[TYPE/]REPO_ID[@REVISION][/PATH]
        [TYPE/]REPO_ID[@REVISION][/PATH]
        hf://buckets/NAMESPACE/NAME[/PREFIX]
        buckets/NAMESPACE/NAME[/PREFIX]

    Where *TYPE* is one of ``models``, ``datasets``, ``spaces`` (mapped to
    the canonical singular form).

    Args:
        url: The URI or identifier to parse.
        default_type: Repo type to assume when none is present in *url*.
            Defaults to ``"model"``.

    Returns:
        :class:`ParsedHfUrl` for repository paths, :class:`ParsedBucketUrl`
        for bucket paths.

    Raises:
        ValueError: If the URL cannot be structurally parsed (e.g. a bucket
            path missing ``namespace/name``).
    """
    # Strip the hf:// prefix if present.
    path = url[len(HF_PREFIX) :] if url.startswith(HF_PREFIX) else url

    if not path:
        raise ValueError(f"Empty path in URL: '{url}'")

    # ------------------------------------------------------------------
    # Detect type from first segment
    # ------------------------------------------------------------------
    first_segment = path.split("/", 1)[0]

    has_explicit_type = first_segment in _HF_TYPES_MAPPING
    if has_explicit_type:
        mapped_type = _HF_TYPES_MAPPING[first_segment]
        path = path.split("/", 1)[1] if "/" in path else ""

        if mapped_type == "bucket":
            return _parse_bucket_path(path, original_url=url)
        repo_type = mapped_type
    else:
        repo_type = default_type

    # ------------------------------------------------------------------
    # Parse repo_id, optional @revision, and path_in_repo
    # ------------------------------------------------------------------
    return _parse_repo_path(path, repo_type=repo_type, has_explicit_type=has_explicit_type)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_bucket_path(path: str, *, original_url: str = "") -> ParsedBucketUrl:
    """Parse the portion after ``buckets/`` into a :class:`ParsedBucketUrl`."""
    parts = path.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid bucket path: '{original_url or path}'. "
            "Expected format: hf://buckets/namespace/bucket_name[/prefix]"
        )
    bucket_id = f"{parts[0]}/{parts[1]}"
    prefix = parts[2] if len(parts) > 2 else ""
    return ParsedBucketUrl(bucket_id=bucket_id, path=prefix)


def _parse_repo_path(path: str, *, repo_type: str, has_explicit_type: bool = False) -> ParsedHfUrl:
    """Parse ``REPO_ID[@REVISION][/PATH]`` into a :class:`ParsedHfUrl`.

    Assumes the type prefix (if any) has already been stripped.
    """
    if not path:
        return ParsedHfUrl(repo_type=repo_type, repo_id="", revision=None, path_in_repo="", has_explicit_type=has_explicit_type)

    # -- Handle @revision syntax ------------------------------------------
    # The "@" can appear in the first two path segments (repo_id part).
    # We only split on "@" if it appears in those first two segments.
    if "@" in "/".join(path.split("/")[:2]):
        repo_id, revision_and_rest = path.split("@", 1)

        if "/" in revision_and_rest:
            # Check for special refs (refs/pr/N, refs/convert/…) that contain "/"
            match = SPECIAL_REFS_REVISION_REGEX.search(revision_and_rest)
            if match is not None:
                revision = match.group()
                path_in_repo = SPECIAL_REFS_REVISION_REGEX.sub("", revision_and_rest).lstrip("/")
            else:
                revision, path_in_repo = revision_and_rest.split("/", 1)
        else:
            revision = revision_and_rest
            path_in_repo = ""

        revision = unquote(revision)
        return ParsedHfUrl(repo_type=repo_type, repo_id=repo_id, revision=revision, path_in_repo=path_in_repo, has_explicit_type=has_explicit_type)

    # -- No @revision: split into repo_id (max 2 segments) + path --------
    parts = path.split("/")
    if len(parts) >= 2:
        repo_id = f"{parts[0]}/{parts[1]}"
        path_in_repo = "/".join(parts[2:])
    else:
        repo_id = parts[0]
        path_in_repo = ""

    return ParsedHfUrl(repo_type=repo_type, repo_id=repo_id, revision=None, path_in_repo=path_in_repo, has_explicit_type=has_explicit_type)
