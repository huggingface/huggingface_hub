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
"""Tests for the centralized HF URI parser ('huggingface_hub.utils._hf_uris')."""

import pytest

from huggingface_hub.errors import HfUriError
from huggingface_hub.utils import HfMount, HfUri, parse_hf_mount, parse_hf_uri


# ---------------------------------------------------------------------------
# HfUri success cases: (uri, expected_HfUri, expected_roundtrip)
# ---------------------------------------------------------------------------
URI_SUCCESS_CASES: list[tuple[str, HfUri, str]] = [
    # --- Models ----------------------------------------------------------------
    # Namespaced model (implicit type prefix)
    (
        "hf://my-org/my-model",
        HfUri(type="model", id="my-org/my-model"),
        "hf://models/my-org/my-model",
    ),
    # Namespaced model (explicit type prefix)
    (
        "hf://models/my-org/my-model",
        HfUri(type="model", id="my-org/my-model"),
        "hf://models/my-org/my-model",
    ),
    # Model with path inside the repo
    (
        "hf://models/my-org/my-model/config.json",
        HfUri(type="model", id="my-org/my-model", path_in_repo="config.json"),
        "hf://models/my-org/my-model/config.json",
    ),
    # Model with explicit revision
    (
        "hf://my-org/my-model@v1.0",
        HfUri(type="model", id="my-org/my-model", revision="v1.0"),
        "hf://models/my-org/my-model@v1.0",
    ),
    # Model with revision and path
    (
        "hf://models/my-org/my-model@dev/sub/file.bin",
        HfUri(type="model", id="my-org/my-model", revision="dev", path_in_repo="sub/file.bin"),
        "hf://models/my-org/my-model@dev/sub/file.bin",
    ),
    # --- Datasets --------------------------------------------------------------
    (
        "hf://datasets/my-org/my-dataset",
        HfUri(type="dataset", id="my-org/my-dataset"),
        "hf://datasets/my-org/my-dataset",
    ),
    # Dataset with revision and path
    (
        "hf://datasets/my-org/my-dataset@v1/train.csv",
        HfUri(type="dataset", id="my-org/my-dataset", revision="v1", path_in_repo="train.csv"),
        "hf://datasets/my-org/my-dataset@v1/train.csv",
    ),
    # --- Spaces ----------------------------------------------------------------
    (
        "hf://spaces/user/my-space",
        HfUri(type="space", id="user/my-space"),
        "hf://spaces/user/my-space",
    ),
    (
        "hf://spaces/user/my-space@main",
        HfUri(type="space", id="user/my-space", revision="main"),
        "hf://spaces/user/my-space@main",
    ),
    # --- Kernels ---------------------------------------------------------------
    (
        "hf://kernels/my-org/my-kernel",
        HfUri(type="kernel", id="my-org/my-kernel"),
        "hf://kernels/my-org/my-kernel",
    ),
    # --- Buckets ---------------------------------------------------------------
    (
        "hf://buckets/my-org/my-bucket",
        HfUri(type="bucket", id="my-org/my-bucket"),
        "hf://buckets/my-org/my-bucket",
    ),
    (
        "hf://buckets/my-org/my-bucket/sub/path",
        HfUri(type="bucket", id="my-org/my-bucket", path_in_repo="sub/path"),
        "hf://buckets/my-org/my-bucket/sub/path",
    ),
    # Trailing slashes are tolerated and stripped.
    (
        "hf://buckets/my-org/my-bucket/",
        HfUri(type="bucket", id="my-org/my-bucket"),
        "hf://buckets/my-org/my-bucket",
    ),
    # --- Special revisions (refs/pr/N, refs/convert/parquet) -------------------
    # Special PR revision: must be matched eagerly even though it contains '/'
    (
        "hf://datasets/foo/bar@refs/pr/10/file.csv",
        HfUri(type="dataset", id="foo/bar", revision="refs/pr/10", path_in_repo="file.csv"),
        "hf://datasets/foo/bar@refs/pr/10/file.csv",
    ),
    # Special convert revision
    (
        "hf://datasets/foo/bar@refs/convert/parquet/data.parquet",
        HfUri(
            type="dataset",
            id="foo/bar",
            revision="refs/convert/parquet",
            path_in_repo="data.parquet",
        ),
        "hf://datasets/foo/bar@refs/convert/parquet/data.parquet",
    ),
    # Convert ref name with hyphen (and dot) — must not be split at the hyphen.
    (
        "hf://datasets/foo/bar@refs/convert/parquet-v2/data.parquet",
        HfUri(
            type="dataset",
            id="foo/bar",
            revision="refs/convert/parquet-v2",
            path_in_repo="data.parquet",
        ),
        "hf://datasets/foo/bar@refs/convert/parquet-v2/data.parquet",
    ),
    (
        "hf://datasets/foo/bar@refs/convert/duckdb.v1/data.db",
        HfUri(
            type="dataset",
            id="foo/bar",
            revision="refs/convert/duckdb.v1",
            path_in_repo="data.db",
        ),
        "hf://datasets/foo/bar@refs/convert/duckdb.v1/data.db",
    ),
    # URL-encoded special revision
    (
        "hf://datasets/foo/bar@refs%2Fpr%2F10/file.csv",
        HfUri(type="dataset", id="foo/bar", revision="refs/pr/10", path_in_repo="file.csv"),
        "hf://datasets/foo/bar@refs/pr/10/file.csv",
    ),
    # Branch name containing '/' (e.g. 'feature/foo'). Must be URL-encoded so that the
    # parser does not split it at the first '/' and treat the rest as a path-in-repo.
    (
        "hf://my-org/my-model@feature%2Ffoo/config.json",
        HfUri(type="model", id="my-org/my-model", revision="feature/foo", path_in_repo="config.json"),
        "hf://models/my-org/my-model@feature%2Ffoo/config.json",
    ),
    # Special revision with no path after it
    (
        "hf://my-org/my-model@refs/pr/3",
        HfUri(type="model", id="my-org/my-model", revision="refs/pr/3"),
        "hf://models/my-org/my-model@refs/pr/3",
    ),
]


# ---------------------------------------------------------------------------
# HfUri failure cases: (uri, error_substring)
# ---------------------------------------------------------------------------
URI_FAILURE_CASES: list[tuple[str, str]] = [
    # Missing protocol
    ("gpt2", "Must start with 'hf://'"),
    ("https://huggingface.co/gpt2", "Must start with 'hf://'"),
    ("hf:/gpt2", "Must start with 'hf://'"),
    # Empty body after protocol
    ("hf://", "Empty body"),
    # Empty repo id (just a type prefix)
    ("hf://datasets", "Missing identifier"),
    ("hf://datasets/", "Missing repository id"),
    ("hf://models/", "Missing repository id"),
    ("hf://buckets", "Missing identifier"),
    ("hf://buckets/", "Bucket id must be 'namespace/name'"),
    # Singular type forms are forbidden
    ("hf://dataset/foo/bar", "Type prefix must be plural"),
    ("hf://model/my-org/my-model", "Type prefix must be plural"),
    ("hf://space/user/my-space", "Type prefix must be plural"),
    ("hf://bucket/org/b", "Type prefix must be plural"),
    # Canonical repos (no namespace) are forbidden
    ("hf://gpt2", "Repository id must be 'namespace/name'"),
    ("hf://models/gpt2", "Repository id must be 'namespace/name'"),
    ("hf://datasets/squad", "Repository id must be 'namespace/name'"),
    ("hf://gpt2@v1", "Repository id must be 'namespace/name'"),
    ("hf://gpt2@v1/config.json", "Repository id must be 'namespace/name'"),
    # Buckets must always have namespace/name
    ("hf://buckets/single-segment", "Bucket id must be 'namespace/name'"),
    # Buckets cannot have a revision
    ("hf://buckets/org/b@v1", "do not support a revision"),
    ("hf://buckets/org/b@v1/path", "do not support a revision"),
    # Empty revision
    ("hf://my-org/my-model@", "Empty revision"),
    ("hf://datasets/foo/bar@/file", "Empty revision"),
    # Empty repo id before '@'
    ("hf://@v1/file", "Missing repository id"),
    # Repo id with too many slashes
    ("hf://a/b/c@v1", "Repository id must be 'namespace/name'"),
    # Invalid repo id chars (validated by validate_repo_id)
    ("hf://datasets/foo/.invalid", "Repo id must use alphanumeric"),
    ("hf://models/foo--bar/baz", "Cannot have -- or .."),
    # Empty path segments (adjacent slashes)
    ("hf://models/org/m//sub", "empty segments"),
    ("hf://buckets/org/b//sub", "empty segments"),
    ("hf://models/org/m/sub//dir", "empty segments"),
    ("hf://models/org/m@main//file.txt", "empty segments"),
    ("hf://datasets/foo/bar@refs/pr/10//file.csv", "empty segments"),
]


# ---------------------------------------------------------------------------
# HfUri direct init invalid cases: (kwargs, error_substring)
# ---------------------------------------------------------------------------
URI_DIRECT_INIT_INVALID_CASES: list[tuple[dict, str]] = [
    ({"type": "unknown", "id": "org/repo"}, "Invalid type"),
    ({"type": "model", "id": "gpt2"}, "namespace/name"),
    ({"type": "model", "id": "a/b/c"}, "namespace/name"),
    ({"type": "dataset", "id": "foo--bar/baz"}, "Cannot have -- or .."),
    ({"type": "model", "id": "org/repo", "revision": ""}, "empty string"),
    ({"type": "bucket", "id": "org/bucket", "revision": "main"}, "do not support a revision"),
    ({"type": "model", "id": "org/repo", "path_in_repo": "a//b"}, "empty segments"),
]


# ---------------------------------------------------------------------------
# HfMount success cases: (mount_str, expected_HfMount, expected_roundtrip)
# ---------------------------------------------------------------------------
MOUNT_SUCCESS_CASES: list[tuple[str, HfMount, str]] = [
    (
        "hf://my-org/my-model:/data",
        HfMount(source=HfUri(type="model", id="my-org/my-model"), mount_path="/data"),
        "hf://models/my-org/my-model:/data",
    ),
    (
        "hf://my-org/my-model:/data:ro",
        HfMount(source=HfUri(type="model", id="my-org/my-model"), mount_path="/data", read_only=True),
        "hf://models/my-org/my-model:/data:ro",
    ),
    (
        "hf://my-org/my-model:/data:rw",
        HfMount(source=HfUri(type="model", id="my-org/my-model"), mount_path="/data", read_only=False),
        "hf://models/my-org/my-model:/data:rw",
    ),
    (
        "hf://datasets/my-org/my-dataset:/mnt",
        HfMount(source=HfUri(type="dataset", id="my-org/my-dataset"), mount_path="/mnt"),
        "hf://datasets/my-org/my-dataset:/mnt",
    ),
    # Mount path with revision
    (
        "hf://datasets/my-org/my-dataset@v1:/mnt:ro",
        HfMount(
            source=HfUri(type="dataset", id="my-org/my-dataset", revision="v1"),
            mount_path="/mnt",
            read_only=True,
        ),
        "hf://datasets/my-org/my-dataset@v1:/mnt:ro",
    ),
    # Mount path with sub-path inside repo
    (
        "hf://datasets/my-org/my-dataset/train:/mnt",
        HfMount(
            source=HfUri(type="dataset", id="my-org/my-dataset", path_in_repo="train"),
            mount_path="/mnt",
        ),
        "hf://datasets/my-org/my-dataset/train:/mnt",
    ),
    # Bucket with mount path and ro/rw
    (
        "hf://buckets/my-org/my-bucket:/mnt:rw",
        HfMount(
            source=HfUri(type="bucket", id="my-org/my-bucket"),
            mount_path="/mnt",
            read_only=False,
        ),
        "hf://buckets/my-org/my-bucket:/mnt:rw",
    ),
    # Bucket with sub-path and mount
    (
        "hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro",
        HfMount(
            source=HfUri(type="bucket", id="my-org/my-bucket", path_in_repo="sub/dir"),
            mount_path="/mnt",
            read_only=True,
        ),
        "hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro",
    ),
    # Mount path with several path segments
    (
        "hf://my-org/my-model:/path/to/mount",
        HfMount(source=HfUri(type="model", id="my-org/my-model"), mount_path="/path/to/mount"),
        "hf://models/my-org/my-model:/path/to/mount",
    ),
]


# ---------------------------------------------------------------------------
# HfMount failure cases: (mount_str, error_substring)
# ---------------------------------------------------------------------------
MOUNT_FAILURE_CASES: list[tuple[str, str]] = [
    # Missing protocol
    ("my-org/my-model:/data", "Must start with 'hf://'"),
    # Missing mount path entirely
    ("hf://my-org/my-model", "Missing mount path"),
    # Mount path that is just '/'
    ("hf://my-org/my-model:/", "Mount path must be a non-empty absolute path"),
    # Read-only flag without a mount path
    ("hf://my-org/my-model:ro", "':ro'/':rw' suffix is only valid"),
    ("hf://my-org/my-model:rw", "':ro'/':rw' suffix is only valid"),
    # Invalid URI inside the mount (canonical repo without namespace)
    ("hf://gpt2:/data", "Repository id must be 'namespace/name'"),
    ("hf://gpt2:/data:ro", "Repository id must be 'namespace/name'"),
]


# ---------------------------------------------------------------------------
# HfMount direct init invalid cases: (kwargs, error_substring)
# ---------------------------------------------------------------------------
MOUNT_DIRECT_INIT_INVALID_CASES: list[tuple[dict, str]] = [
    ({"source": HfUri(type="model", id="org/repo"), "mount_path": "relative"}, "absolute path"),
    ({"source": HfUri(type="model", id="org/repo"), "mount_path": "/"}, "absolute path"),
]


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.parametrize(("uri", "expected", "expected_roundtrip"), URI_SUCCESS_CASES)
def test_parse_hf_uri_success(uri: str, expected: HfUri, expected_roundtrip: str) -> None:
    result = parse_hf_uri(uri)
    assert result == expected
    assert result.to_uri() == expected_roundtrip
    # Re-parsing the canonical form must yield the same URI (idempotency).
    assert parse_hf_uri(expected_roundtrip) == expected


@pytest.mark.parametrize(("uri", "error_substring"), URI_FAILURE_CASES)
def test_parse_hf_uri_failure(uri: str, error_substring: str) -> None:
    with pytest.raises(HfUriError, match=error_substring) as exc_info:
        parse_hf_uri(uri)
    assert exc_info.value.uri == uri


@pytest.mark.parametrize(("kwargs", "error_substring"), URI_DIRECT_INIT_INVALID_CASES)
def test_hf_uri_direct_init_invalid(kwargs: dict, error_substring: str) -> None:
    with pytest.raises(HfUriError, match=error_substring):
        HfUri(**kwargs)


@pytest.mark.parametrize(("mount_str", "expected", "expected_roundtrip"), MOUNT_SUCCESS_CASES)
def test_parse_hf_mount_success(mount_str: str, expected: HfMount, expected_roundtrip: str) -> None:
    result = parse_hf_mount(mount_str)
    assert result == expected
    assert result.to_uri() == expected_roundtrip
    # Re-parsing the canonical form must yield the same mount (idempotency).
    assert parse_hf_mount(expected_roundtrip) == expected


@pytest.mark.parametrize(("mount_str", "error_substring"), MOUNT_FAILURE_CASES)
def test_parse_hf_mount_failure(mount_str: str, error_substring: str) -> None:
    with pytest.raises(HfUriError, match=error_substring) as exc_info:
        parse_hf_mount(mount_str)
    assert exc_info.value.uri == mount_str


@pytest.mark.parametrize(("kwargs", "error_substring"), MOUNT_DIRECT_INIT_INVALID_CASES)
def test_hf_mount_direct_init_invalid(kwargs: dict, error_substring: str) -> None:
    with pytest.raises(HfUriError, match=error_substring):
        HfMount(**kwargs)
