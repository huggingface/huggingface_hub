# Copyright 2026-present, the HuggingFace Inc. team.
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
"""Centralized parser for Hugging Face Hub URIs ('hf://...').

A HF URI is a URI-like string that identifies a location on the Hugging Face
Hub: a model/dataset/space/kernel repository, a bucket, optionally a revision,
optionally a path inside the repo or bucket, and optionally a local mount path
with a ':ro'/':rw' flag (used by Spaces and Jobs volumes).

Canonical syntax:

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>][:<MOUNT_PATH>[:ro|:rw]]
```

See 'docs/source/en/package_reference/hf_uris.md' for the full grammar and examples.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote

from huggingface_hub import constants
from huggingface_hub.errors import HfUriError, HFValidationError

from ._validators import validate_repo_id


# Inverse map (singular → plural URI prefix). Built once from the canonical
# 'constants.HF_URI_TYPE_PREFIXES' and used to render URIs.
_TYPE_TO_PREFIX: dict[constants.HfUriType, str] = {v: k for k, v in constants.HF_URI_TYPE_PREFIXES.items()}

# Same map but typed as 'dict[str, str]' so it can be indexed with arbitrary
# strings without confusing the type-checker (used only to suggest the plural
# form when the user wrote a singular type like 'hf://dataset/...').
_PLURAL_FROM_SINGULAR_NAME: dict[str, str] = {str(k): v for k, v in _TYPE_TO_PREFIX.items()}

# Special revisions that contain a '/'. They take precedence when splitting
# the part after '@' into '<revision>/<path-in-repo>'. Matches 'refs/pr/N'
# (Pull Request refs) and 'refs/convert/<name>' (e.g. parquet conversions).
# The conversion name allows the typical git ref characters '[a-zA-Z0-9_.-]'
# so names like 'parquet-v2' or 'duckdb.v1' round-trip correctly.
_SPECIAL_REFS_REVISION_REGEX = re.compile(r"^refs/(?:convert/[\w.-]+|pr/\d+)")


@dataclass(frozen=True)
class HfUri:
    """Parsed representation of a Hugging Face Hub URI ('hf://...').

    Attributes:
        type (`str`):
            One of 'model', 'dataset', 'space', 'kernel' or 'bucket'.
        id (`str`):
            The repository id (e.g. 'gpt2' or 'my-org/my-model') for repo URIs, or the bucket id (always 'namespace/name') for bucket URIs.
        revision (`str`, *optional*):
            The revision specified after '@' in the URI, URL-decoded. 'None' if no revision was specified, or for bucket URIs (which
            never carry a revision). Special refs like 'refs/pr/10' and 'refs/convert/parquet' are preserved as-is.
        path_in_repo (`str`):
            The path inside the repo or bucket. Empty string if the URI points at the root.
        mount_path (`str`, *optional*):
            The local mount path specified after ':' (always starts with '/'). 'None' if the URI is a plain location URI.
        read_only (`bool`, *optional*):
            True if the URI ends with ':ro', False if it ends with ':rw', 'None' if no read/write flag was provided.
    """

    type: constants.HfUriType
    id: str
    revision: str | None = None
    path_in_repo: str = ""
    mount_path: str | None = None
    read_only: bool | None = None

    @property
    def is_bucket(self) -> bool:
        """True if this URI points at a bucket."""
        return self.type == "bucket"

    @property
    def is_repo(self) -> bool:
        """True if this URI points at a repository (model, dataset, space or kernel)."""
        return self.type != "bucket"

    def __str__(self) -> str:
        return self.to_uri()

    def to_uri(self) -> str:
        """Render the URI as a canonical 'hf://' string.

        The type prefix is always written explicitly (e.g. 'hf://models/gpt2', not 'hf://gpt2').
        """
        parts: list[str] = [constants.HF_PROTOCOL, _TYPE_TO_PREFIX[self.type], "/", self.id]
        if self.revision is not None:
            # Encode '/' as '%2F' for revisions that would otherwise be split as '<revision>/<path>'
            # at parse time. Special refs ('refs/pr/N', 'refs/convert/<name>') are kept verbatim
            # because the parser matches them eagerly.
            revision = self.revision
            if "/" in revision and _SPECIAL_REFS_REVISION_REGEX.fullmatch(revision) is None:
                revision = revision.replace("/", "%2F")
            parts.append(f"@{revision}")
        if self.path_in_repo:
            parts.append(f"/{self.path_in_repo}")
        if self.mount_path is not None:
            parts.append(f":{self.mount_path}")
            if self.read_only is True:
                parts.append(":ro")
            elif self.read_only is False:
                parts.append(":rw")
        return "".join(parts)


def parse_hf_uri(uri: str) -> HfUri:
    """Parse a Hugging Face Hub URI ('hf://...').

    A HF URI is a URI-like string identifying a location on the Hugging Face Hub. The full grammar is:

    ```
    hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>][:<MOUNT_PATH>[:ro|:rw]]
    ```

    See 'docs/source/en/package_reference/hf_uris.md' for the full specification.

    Args:
        uri (`str`):
            The URI to parse. Must start with 'hf://'.

    Returns:
        [`HfUri`]: the parsed URI.

    Raises:
        [`HfUriError`]:
            If the URI is malformed (missing prefix, invalid type, missing id, etc.).

    Examples:
        ```py
        >>> from huggingface_hub.utils import parse_hf_uri
        >>> parse_hf_uri("hf://gpt2")
        HfUri(type='model', id='gpt2', revision=None, path_in_repo='', mount_path=None, read_only=None)
        >>> parse_hf_uri("hf://datasets/squad@refs/pr/3/train.json")
        HfUri(type='dataset', id='squad', revision='refs/pr/3', path_in_repo='train.json', mount_path=None, read_only=None)
        >>> parse_hf_uri("hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro")
        HfUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir', mount_path='/mnt', read_only=True)
        ```
    """
    if not uri.startswith(constants.HF_PROTOCOL):
        raise HfUriError(
            f"Invalid HF URI '{uri}': must start with '{constants.HF_PROTOCOL}'. "
            f"Expected format: {constants.HF_PROTOCOL}[<TYPE>/]<ID>[@<REVISION>][/<PATH>][:<MOUNT_PATH>[:ro|:rw]]"
        )

    raw = uri
    body = uri[len(constants.HF_PROTOCOL) :]
    if not body:
        raise HfUriError(f"Invalid HF URI '{raw}': empty body after '{constants.HF_PROTOCOL}'.")

    location, mount_path, read_only = _split_mount(body, raw=raw)
    type_, location = _split_type(location, raw=raw)

    if type_ == "bucket":
        return _parse_bucket_body(location, type_, mount_path, read_only, raw=raw)
    return _parse_repo_body(location, type_, mount_path, read_only, raw=raw)


def _split_mount(body: str, *, raw: str) -> tuple[str, str | None, bool | None]:
    """Split the ':<MOUNT_PATH>[:ro|:rw]' suffix from 'body'.

    Returns '(location, mount_path, read_only)' where 'mount_path' is 'None' if no mount segment is present.
    """
    read_only: bool | None = None
    if body.endswith(":ro"):
        read_only = True
        body = body[:-3]
    elif body.endswith(":rw"):
        read_only = False
        body = body[:-3]

    # Mount paths always start with '/', so the delimiter is ':/'.
    # We use rfind() because the mount segment is always trailing
    idx = body.rfind(":/")
    if idx == -1:
        if read_only is not None:
            raise HfUriError(
                f"Invalid HF URI '{raw}': ':ro'/':rw' suffix is only valid when a mount path is provided "
                "(e.g. 'hf://...:/<MOUNT_PATH>:ro')."
            )
        return body, None, None

    location = body[:idx]
    mount_path = body[idx + 1 :]  # includes the leading '/'
    if not location:
        raise HfUriError(f"Invalid HF URI '{raw}': missing location before mount path.")
    if not mount_path.startswith("/") or mount_path == "/":
        raise HfUriError(
            f"Invalid HF URI '{raw}': mount path must be a non-empty absolute path starting with '/', got '{mount_path}'."
        )
    return location, mount_path, read_only


def _split_type(location: str, *, raw: str) -> tuple[constants.HfUriType, str]:
    """Detect the (optional) type prefix and return '(type, remaining_location)'.

    A missing type prefix defaults to 'model'. Singular forms ('model/', 'dataset/', etc.) are explicitly rejected with a helpful error.
    """
    slash_idx = location.find("/")
    if slash_idx == -1:
        # Single segment, no prefix. Reject if it looks like a bare type name.
        if location in constants.HF_URI_TYPE_PREFIXES:
            raise HfUriError(
                f"Invalid HF URI '{raw}': missing identifier after '{location}'. Expected '{constants.HF_PROTOCOL}{location}/<ID>'."
            )
        if (singular_plural := _PLURAL_FROM_SINGULAR_NAME.get(location)) is not None:
            raise HfUriError(
                f"Invalid HF URI '{raw}': type prefix must be plural. Did you mean '{constants.HF_PROTOCOL}{singular_plural}/...'?"
            )
        return "model", location

    first = location[:slash_idx]
    rest = location[slash_idx + 1 :]
    if first in constants.HF_URI_TYPE_PREFIXES:
        return constants.HF_URI_TYPE_PREFIXES[first], rest
    if (singular_plural := _PLURAL_FROM_SINGULAR_NAME.get(first)) is not None:
        raise HfUriError(
            f"Invalid HF URI '{raw}': type prefix must be plural, got '{first}/'. Did you mean '{singular_plural}/'?"
        )
    return "model", location


def _parse_bucket_body(
    location: str,
    type_: constants.HfUriType,
    mount_path: str | None,
    read_only: bool | None,
    *,
    raw: str,
) -> HfUri:
    """Parse the body of a bucket URI: 'namespace/name[/path]'."""
    if "@" in location:
        raise HfUriError(f"Invalid HF URI '{raw}': bucket URIs do not support a revision marker ('@').")
    location = location.strip("/")
    parts = location.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise HfUriError(f"Invalid HF URI '{raw}': bucket id must be 'namespace/name', got '{location}'.")
    bucket_id = f"{parts[0]}/{parts[1]}"
    path_in_bucket = parts[2].strip("/") if len(parts) >= 3 else ""
    return HfUri(
        type=type_,
        id=bucket_id,
        revision=None,
        path_in_repo=path_in_bucket,
        mount_path=mount_path,
        read_only=read_only,
    )


def _parse_repo_body(
    location: str,
    type_: constants.HfUriType,
    mount_path: str | None,
    read_only: bool | None,
    *,
    raw: str,
) -> HfUri:
    """Parse the body of a repo URI: '<repo_id>[@<revision>][/<path>]'."""
    location = location.strip("/")
    if not location:
        raise HfUriError(f"Invalid HF URI '{raw}': missing repository id.")

    # The first '@' separates the repo_id from the revision (and rest of path).
    # No valid repo_id contains '@' and no valid revision contains '@'.
    at_idx = location.find("@")
    revision: str | None
    if at_idx == -1:
        # No revision. Take the first 1-2 segments as repo_id, rest as path_in_repo.
        revision = None
        parts = location.split("/", 2)
        if len(parts) == 1:
            repo_id = parts[0]
            path_in_repo = ""
        else:
            repo_id = f"{parts[0]}/{parts[1]}"
            path_in_repo = parts[2] if len(parts) > 2 else ""
    else:
        repo_id = location[:at_idx]
        rev_and_path = location[at_idx + 1 :]
        if not repo_id:
            raise HfUriError(f"Invalid HF URI '{raw}': missing repository id before '@'.")
        if repo_id.count("/") > 1:
            raise HfUriError(
                f"Invalid HF URI '{raw}': repository id must be 'name' or 'namespace/name', got '{repo_id}'."
            )
        # Special refs like 'refs/pr/10' contain '/' and must be matched eagerly,
        # otherwise we would split them at the first '/' and treat the rest as a path.
        match = _SPECIAL_REFS_REVISION_REGEX.match(rev_and_path)
        if match is not None:
            revision = match.group()
            path_in_repo = rev_and_path[len(revision) :].lstrip("/")
        else:
            slash_idx = rev_and_path.find("/")
            if slash_idx == -1:
                revision = rev_and_path
                path_in_repo = ""
            else:
                revision = rev_and_path[:slash_idx]
                path_in_repo = rev_and_path[slash_idx + 1 :]
        revision = unquote(revision)
        if not revision:
            raise HfUriError(f"Invalid HF URI '{raw}': empty revision after '@'.")

    try:
        validate_repo_id(repo_id)
    except HFValidationError as e:
        raise HfUriError(f"Invalid HF URI '{raw}': {e}") from e

    return HfUri(
        type=type_,
        id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        mount_path=mount_path,
        read_only=read_only,
    )
