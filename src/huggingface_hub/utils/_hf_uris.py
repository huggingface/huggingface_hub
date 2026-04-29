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
"""Centralized parser for Hugging Face Hub URIs ('hf://...') and mount specifications.

A HF URI is a URI-like string that identifies a location on the Hugging Face
Hub: a model/dataset/space/kernel repository, a bucket, optionally a revision,
and optionally a path inside the repo or bucket.

Canonical syntax:

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
```

A HF mount wraps a HF URI with a local mount path and an optional ':ro'/':rw'
flag (used by Spaces and Jobs volumes):

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
```

See 'docs/source/en/package_reference/hf_uris.md' for the full grammar and examples.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import unquote

from huggingface_hub import constants
from huggingface_hub.errors import HfUriError, HFValidationError

from ._validators import validate_repo_id


# Inverse map (singular -> plural URI prefix). Built once from the canonical
# 'constants.HF_URI_TYPE_PREFIXES' and used to render URIs.
_TYPE_TO_PREFIX: dict[str, str] = {v: k for k, v in constants.HF_URI_TYPE_PREFIXES.items()}

# Special revisions that contain a '/'. They take precedence when splitting
# the part after '@' into '<revision>/<path-in-repo>'. Matches 'refs/pr/N'
# (Pull Request refs) and 'refs/convert/<name>' (e.g. parquet conversions).
# The conversion name allows the typical git ref characters '[a-zA-Z0-9_.-]'
# so names like 'parquet-v2' or 'duckdb.v1' round-trip correctly.
_SPECIAL_REFS_REVISION_REGEX = re.compile(r"^refs/(?:convert/[\w.-]+|pr/\d+)")

# Same as constants.HfUriType, but as a set of strings for easy lookup.)
_VALID_URI_TYPES: frozenset[str] = frozenset(constants.HF_URI_TYPE_PREFIXES.values())


@dataclass(frozen=True)
class HfUri:
    """Parsed representation of a Hugging Face Hub URI ('hf://...').

    Attributes:
        type (`str`):
            One of 'model', 'dataset', 'space', 'kernel' or 'bucket'.
        id (`str`):
            The repository id ('namespace/name', e.g. 'my-org/my-model') for repo URIs, or the bucket id ('namespace/name') for bucket URIs.
        revision (`str`, *optional*):
            The revision specified after '@' in the URI, URL-decoded. 'None' if no revision was specified, or for bucket URIs (which
            never carry a revision). Special refs like 'refs/pr/10' and 'refs/convert/parquet' are preserved as-is.
        path_in_repo (`str`):
            The path inside the repo or bucket. Empty string if the URI points at the root.
    """

    type: constants.HfUriType
    id: str
    revision: str | None = None
    path_in_repo: str = ""
    _raw: str | None = field(repr=False, hash=False, compare=False, default=None)

    def __post_init__(self) -> None:
        uri = self._raw or ""  # For error messages

        # Check valid URI type
        if self.type not in _VALID_URI_TYPES:
            raise HfUriError(uri=uri, msg=f"Invalid type '{self.type}'. Must be one of {sorted(_VALID_URI_TYPES)}.")

        # Check valid ID
        if not self.id or self.id.count("/") != 1:
            raise HfUriError(uri=uri, msg=f"Id must be 'namespace/name', got '{self.id}'.")
        if self.type != "bucket":
            try:
                validate_repo_id(self.id)
            except HFValidationError as e:
                raise HfUriError(uri=uri, msg=str(e)) from e

        # Check valid revision
        if self.revision is not None and not self.revision:
            raise HfUriError(uri=uri, msg="Revision must not be an empty string.")
        if self.type == "bucket" and self.revision is not None:
            raise HfUriError(uri=uri, msg="Bucket URIs do not support a revision.")

        # Check valid path in repo
        if self.path_in_repo:
            if self.path_in_repo.startswith("/") or "//" in self.path_in_repo:
                raise HfUriError(uri=uri, msg=f"Path must not contain empty segments (got '{self.path_in_repo}').")

    @property
    def is_bucket(self) -> bool:
        """True if this URI points at a bucket."""
        return self.type == "bucket"

    @property
    def is_repo(self) -> bool:
        """True if this URI points at a repository (model, dataset, space or kernel)."""
        return self.type != "bucket"

    def to_uri(self) -> str:
        """Render the URI as a canonical 'hf://' string.

        The type prefix is always written explicitly (e.g. 'hf://models/my-org/my-model').
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
        return "".join(parts)


@dataclass(frozen=True)
class HfMount:
    """A HF URI paired with a local mount path and optional read-only flag.

    Used by Spaces and Jobs to describe volume mounts. The full syntax is:

    ```
    hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
    ```

    Attributes:
        source ([`HfUri`]):
            The parsed HF URI identifying the Hub resource to mount.
        mount_path (`str`):
            The local mount path (always starts with '/').
        read_only (`bool`, *optional*):
            True if the mount ends with ':ro', False if it ends with ':rw', 'None' if no flag was provided.
    """

    source: HfUri
    mount_path: str
    read_only: bool | None = None
    _raw: str | None = field(repr=False, hash=False, compare=False, default=None)

    def __post_init__(self) -> None:
        raw = self._raw or ""
        if not self.mount_path.startswith("/") or self.mount_path == "/":
            raise HfUriError(
                uri=raw,
                msg=f"Mount path must be a non-empty absolute path starting with '/', got '{self.mount_path}'.",
            )

    def to_uri(self) -> str:
        """Render the mount as a canonical 'hf://' string.

        Example: 'hf://models/my-org/my-model:/data:ro'
        """
        parts = [self.source.to_uri(), ":", self.mount_path]
        if self.read_only is not None:
            parts.append(":ro" if self.read_only else ":rw")
        return "".join(parts)


def parse_hf_uri(uri: str) -> HfUri:
    """Parse a Hugging Face Hub URI ('hf://...').

    A HF URI is a URI-like string identifying a location on the Hugging Face Hub. The full grammar is:

    ```
    hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
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
        >>> parse_hf_uri("hf://my-org/my-model")
        HfUri(type='model', id='my-org/my-model', revision=None, path_in_repo='')
        >>> parse_hf_uri("hf://datasets/my-org/my-dataset@refs/pr/3/train.json")
        HfUri(type='dataset', id='my-org/my-dataset', revision='refs/pr/3', path_in_repo='train.json')
        ```
    """
    if not uri.startswith(constants.HF_PROTOCOL):
        raise HfUriError(
            uri,
            f"Must start with '{constants.HF_PROTOCOL}'. "
            f"Expected format: {constants.HF_PROTOCOL}[<TYPE>/]<ID>[@<REVISION>][/<PATH>]",
        )

    raw = uri
    body = uri[len(constants.HF_PROTOCOL) :]
    if not body:
        raise HfUriError(uri, f"Empty body after '{constants.HF_PROTOCOL}'.")

    type_, location = _split_type(body, raw=raw)

    if type_ == "bucket":
        return _parse_bucket_body(location, type_, raw=raw)
    return _parse_repo_body(location, type_, raw=raw)


def parse_hf_mount(mount_str: str) -> HfMount:
    """Parse a HF mount specification ('hf://...:<MOUNT_PATH>[:ro|:rw]').

    A mount specification is a HF URI followed by a local mount path and an optional read-only/read-write flag.
    The full grammar is:

    ```
    hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
    ```

    See 'docs/source/en/package_reference/hf_uris.md' for the full specification.

    Args:
        mount_str (`str`):
            The mount string to parse. Must start with 'hf://' and contain a ':<MOUNT_PATH>' segment.

    Returns:
        [`HfMount`]: the parsed mount.

    Raises:
        [`HfUriError`]:
            If the mount string is malformed (missing mount path, invalid URI, etc.).

    Examples:
        ```py
        >>> from huggingface_hub.utils import parse_hf_mount
        >>> parse_hf_mount("hf://my-org/my-model:/data:ro")
        HfMount(source=HfUri(type='model', id='my-org/my-model', revision=None, path_in_repo=''), mount_path='/data', read_only=True)
        >>> parse_hf_mount("hf://buckets/my-org/my-bucket/sub/dir:/mnt:rw")
        HfMount(source=HfUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir'), mount_path='/mnt', read_only=False)
        ```
    """
    if not mount_str.startswith(constants.HF_PROTOCOL):
        raise HfUriError(
            uri=mount_str,
            msg=f"Must start with '{constants.HF_PROTOCOL}'.",
        )

    raw = mount_str
    body = mount_str[len(constants.HF_PROTOCOL) :]
    if not body:
        raise HfUriError(uri=raw, msg=f"Empty body after '{constants.HF_PROTOCOL}'.")

    location, mount_path, read_only = _split_mount(body, raw=raw)

    if mount_path is None:
        raise HfUriError(uri=raw, msg="Missing mount path. Expected ':<MOUNT_PATH>' (e.g. 'hf://org/model:/data').")

    # Re-assemble the URI part and parse it
    uri_str = constants.HF_PROTOCOL + location
    try:
        source = parse_hf_uri(uri_str)
    except HfUriError as e:
        raise HfUriError(uri=raw, msg=e.msg) from e

    return HfMount(source=source, mount_path=mount_path, read_only=read_only, _raw=raw)


def _split_mount(body: str, *, raw: str) -> tuple[str, str | None, bool | None]:
    """Split the ':<MOUNT_PATH>[:ro|:rw]' suffix from 'body'.

    Returns '(location, mount_path, read_only)' where 'mount_path' is 'None' if no mount segment is present.
    """
    if body.endswith(":ro"):
        read_only, body = True, body.removesuffix(":ro")
    elif body.endswith(":rw"):
        read_only, body = False, body.removesuffix(":rw")
    else:
        read_only = None

    # Mount paths always start with '/', so the delimiter is ':/'.
    # We use rfind() because the mount segment is always trailing
    idx = body.rfind(":/")
    if idx == -1:
        if read_only is not None:
            raise HfUriError(
                uri=raw,
                msg="':ro'/':rw' suffix is only valid when a mount path is provided (e.g. 'hf://...:/<MOUNT_PATH>:ro').",
            )
        return body, None, None

    location = body[:idx]
    mount_path = body[idx + 1 :]  # includes the leading '/'
    if not location:
        raise HfUriError(uri=raw, msg="Missing location before mount path.")
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
                uri=raw,
                msg=f"Missing identifier after '{location}'. Expected '{constants.HF_PROTOCOL}{location}/<ID>'.",
            )
        if (singular_plural := _TYPE_TO_PREFIX.get(location)) is not None:
            raise HfUriError(
                uri=raw,
                msg=f"Type prefix must be plural. Did you mean '{constants.HF_PROTOCOL}{singular_plural}/...'?",
            )
        return "model", location

    first = location[:slash_idx]
    rest = location[slash_idx + 1 :]
    if first in constants.HF_URI_TYPE_PREFIXES:
        return constants.HF_URI_TYPE_PREFIXES[first], rest
    if (singular_plural := _TYPE_TO_PREFIX.get(first)) is not None:
        raise HfUriError(
            uri=raw, msg=f"Type prefix must be plural, got '{first}/'. Did you mean '{singular_plural}/'?"
        )
    return "model", location


def _parse_bucket_body(
    location: str,
    type_: constants.HfUriType,
    *,
    raw: str,
) -> HfUri:
    """Parse the body of a bucket URI: 'namespace/name[/path]'."""
    if "@" in location:
        raise HfUriError(uri=raw, msg="Bucket URIs do not support a revision marker ('@').")
    location = location.strip("/")
    parts = location.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise HfUriError(uri=raw, msg=f"Bucket id must be 'namespace/name', got '{location}'.")
    bucket_id = f"{parts[0]}/{parts[1]}"
    path_in_bucket = parts[2] if len(parts) >= 3 else ""
    return HfUri(
        type=type_,
        id=bucket_id,
        revision=None,
        path_in_repo=path_in_bucket,
        _raw=raw,
    )


def _parse_repo_body(
    location: str,
    type_: constants.HfUriType,
    *,
    raw: str,
) -> HfUri:
    """Parse the body of a repo URI: '<repo_id>[@<revision>][/<path>]'."""
    location = location.strip("/")
    if not location:
        raise HfUriError(uri=raw, msg="Missing repository id.")

    # The first '@' separates the repo_id from the revision (and rest of path).
    # No valid repo_id contains '@' and no valid revision contains '@'.
    at_idx = location.find("@")
    revision: str | None
    if at_idx == -1:
        # No revision. Take the first 2 segments as repo_id, rest as path_in_repo.
        revision = None
        parts = location.split("/", 2)
        if len(parts) < 2:
            raise HfUriError(uri=raw, msg=f"Repository id must be 'namespace/name', got '{location}'. ")
        repo_id = f"{parts[0]}/{parts[1]}"
        path_in_repo = parts[2] if len(parts) > 2 else ""
    else:
        repo_id = location[:at_idx]
        rev_and_path = location[at_idx + 1 :]
        if not repo_id:
            raise HfUriError(uri=raw, msg="Missing repository id before '@'.")
        if repo_id.count("/") != 1:
            raise HfUriError(uri=raw, msg=f"Repository id must be 'namespace/name', got '{repo_id}'.")
        # Special refs like 'refs/pr/10' contain '/' and must be matched eagerly,
        # otherwise we would split them at the first '/' and treat the rest as a path.
        match = _SPECIAL_REFS_REVISION_REGEX.match(rev_and_path)
        if match is not None:
            revision = match.group()
            path_in_repo = rev_and_path[len(revision) :].removeprefix("/")
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
            raise HfUriError(uri=raw, msg="Empty revision after '@'.")

    return HfUri(
        type=type_,
        id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        _raw=raw,
    )
