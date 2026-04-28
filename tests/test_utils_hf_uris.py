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
"""Tests for the centralized HF URI parser (``huggingface_hub.utils._hf_uris``)."""

from __future__ import annotations

import pytest

from huggingface_hub.utils import HfUri, parse_hf_uri


# A "success" case is described as ``(uri, expected_HfUri, expected_roundtrip)``.
# - ``uri``: the input string fed to ``parse_hf_uri``.
# - ``expected_HfUri``: the expected parsed value.
# - ``expected_roundtrip``: the expected output of ``HfUri.to_string()``. May
#   differ from the input when the URI is reformatted (implicit ``models/``
#   prefix added, URL-encoded revision decoded, ...).
SUCCESS_CASES: list[tuple[str, HfUri, str]] = [
    # --- Models ----------------------------------------------------------------
    # Canonical model id, no namespace
    (
        "hf://gpt2",
        HfUri(type="model", id="gpt2"),
        "hf://models/gpt2",
    ),
    # Canonical model id with explicit ``models/`` prefix
    (
        "hf://models/gpt2",
        HfUri(type="model", id="gpt2"),
        "hf://models/gpt2",
    ),
    # Namespaced model
    (
        "hf://my-org/my-model",
        HfUri(type="model", id="my-org/my-model"),
        "hf://models/my-org/my-model",
    ),
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
    # Canonical model with path
    (
        "hf://gpt2@main/config.json",
        HfUri(type="model", id="gpt2", revision="main", path_in_repo="config.json"),
        "hf://models/gpt2@main/config.json",
    ),
    # --- Datasets --------------------------------------------------------------
    # Canonical dataset (single-segment id, e.g. ``squad``, ``glue``)
    (
        "hf://datasets/squad",
        HfUri(type="dataset", id="squad"),
        "hf://datasets/squad",
    ),
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
    # URL-encoded special revision
    (
        "hf://datasets/foo/bar@refs%2Fpr%2F10/file.csv",
        HfUri(type="dataset", id="foo/bar", revision="refs/pr/10", path_in_repo="file.csv"),
        "hf://datasets/foo/bar@refs/pr/10/file.csv",
    ),
    # Special revision with no path after it
    (
        "hf://gpt2@refs/pr/3",
        HfUri(type="model", id="gpt2", revision="refs/pr/3"),
        "hf://models/gpt2@refs/pr/3",
    ),
    # --- Mount path + ro/rw ----------------------------------------------------
    (
        "hf://gpt2:/data",
        HfUri(type="model", id="gpt2", mount_path="/data"),
        "hf://models/gpt2:/data",
    ),
    (
        "hf://gpt2:/data:ro",
        HfUri(type="model", id="gpt2", mount_path="/data", read_only=True),
        "hf://models/gpt2:/data:ro",
    ),
    (
        "hf://gpt2:/data:rw",
        HfUri(type="model", id="gpt2", mount_path="/data", read_only=False),
        "hf://models/gpt2:/data:rw",
    ),
    (
        "hf://datasets/my-org/my-dataset:/mnt",
        HfUri(type="dataset", id="my-org/my-dataset", mount_path="/mnt"),
        "hf://datasets/my-org/my-dataset:/mnt",
    ),
    # Mount path with revision
    (
        "hf://datasets/my-org/my-dataset@v1:/mnt:ro",
        HfUri(
            type="dataset",
            id="my-org/my-dataset",
            revision="v1",
            mount_path="/mnt",
            read_only=True,
        ),
        "hf://datasets/my-org/my-dataset@v1:/mnt:ro",
    ),
    # Mount path with sub-path inside repo
    (
        "hf://datasets/my-org/my-dataset/train:/mnt",
        HfUri(
            type="dataset",
            id="my-org/my-dataset",
            path_in_repo="train",
            mount_path="/mnt",
        ),
        "hf://datasets/my-org/my-dataset/train:/mnt",
    ),
    # Bucket with mount path and ro/rw
    (
        "hf://buckets/my-org/my-bucket:/mnt:rw",
        HfUri(
            type="bucket",
            id="my-org/my-bucket",
            mount_path="/mnt",
            read_only=False,
        ),
        "hf://buckets/my-org/my-bucket:/mnt:rw",
    ),
    # Bucket with sub-path and mount
    (
        "hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro",
        HfUri(
            type="bucket",
            id="my-org/my-bucket",
            path_in_repo="sub/dir",
            mount_path="/mnt",
            read_only=True,
        ),
        "hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro",
    ),
    # Mount path with several path segments
    (
        "hf://gpt2:/path/to/mount",
        HfUri(type="model", id="gpt2", mount_path="/path/to/mount"),
        "hf://models/gpt2:/path/to/mount",
    ),
]


# A "failure" case is ``(uri, error_substring)``: ``parse_hf_uri(uri)`` must
# raise a ``ValueError`` whose message contains ``error_substring``.
FAILURE_CASES: list[tuple[str, str]] = [
    # Missing protocol
    ("gpt2", "must start with 'hf://'"),
    ("https://huggingface.co/gpt2", "must start with 'hf://'"),
    ("hf:/gpt2", "must start with 'hf://'"),
    # Empty body after protocol
    ("hf://", "empty body"),
    # Empty repo id (just a type prefix)
    ("hf://datasets", "missing identifier"),
    ("hf://datasets/", "missing repository id"),
    ("hf://models/", "missing repository id"),
    ("hf://buckets", "missing identifier"),
    ("hf://buckets/", "bucket id must be 'namespace/name'"),
    # Singular type forms are forbidden
    ("hf://dataset/foo/bar", "must be plural"),
    ("hf://model/gpt2", "must be plural"),
    ("hf://space/user/my-space", "must be plural"),
    ("hf://bucket/org/b", "must be plural"),
    # Buckets must always have namespace/name
    ("hf://buckets/single-segment", "bucket id must be 'namespace/name'"),
    # Buckets cannot have a revision
    ("hf://buckets/org/b@v1", "do not support a revision"),
    ("hf://buckets/org/b@v1/path", "do not support a revision"),
    # Empty revision
    ("hf://gpt2@", "empty revision"),
    ("hf://datasets/foo/bar@/file", "empty revision"),
    # Empty repo id before '@'
    ("hf://@v1/file", "missing repository id"),
    # Repo id with too many slashes
    ("hf://a/b/c@v1", "repository id must be 'name' or 'namespace/name'"),
    # Invalid repo id chars (validated by validate_repo_id)
    ("hf://datasets/foo/.invalid", "Repo id must use alphanumeric"),
    ("hf://models/foo--bar", "Cannot have -- or .."),
    # Mount path that is not absolute
    ("hf://gpt2:/", "mount path must be a non-empty absolute path"),
    # Read-only flag without a mount path
    ("hf://gpt2:ro", "':ro'/':rw' suffix is only valid"),
    ("hf://gpt2:rw", "':ro'/':rw' suffix is only valid"),
    # Wrong type
    ("hf://", "empty body"),
]


@pytest.mark.parametrize(("uri", "expected", "expected_roundtrip"), SUCCESS_CASES)
def test_parse_hf_uri_success(uri: str, expected: HfUri, expected_roundtrip: str) -> None:
    """``parse_hf_uri`` returns the expected ``HfUri`` and round-trips back to a canonical string."""
    result = parse_hf_uri(uri)
    assert result == expected
    assert result.to_string() == expected_roundtrip
    # Re-parsing the canonical form must yield the same URI (idempotency).
    assert parse_hf_uri(expected_roundtrip) == expected


@pytest.mark.parametrize(("uri", "error_substring"), FAILURE_CASES)
def test_parse_hf_uri_failure(uri: str, error_substring: str) -> None:
    """``parse_hf_uri`` rejects malformed URIs with a helpful error message."""
    with pytest.raises(ValueError, match=error_substring):
        parse_hf_uri(uri)


# --- A few targeted tests not easily expressed as parametrized cases. ---------


def test_repo_id_property_on_bucket_uri_raises() -> None:
    uri = parse_hf_uri("hf://buckets/org/b")
    with pytest.raises(AttributeError, match="bucket_id"):
        uri.repo_id  # noqa: B018


def test_bucket_id_property_on_repo_uri_raises() -> None:
    uri = parse_hf_uri("hf://datasets/org/ds")
    with pytest.raises(AttributeError, match="repo_id"):
        uri.bucket_id  # noqa: B018


def test_repo_type_property_on_bucket_uri_raises() -> None:
    uri = parse_hf_uri("hf://buckets/org/b")
    with pytest.raises(AttributeError, match="repo_type"):
        uri.repo_type  # noqa: B018


def test_is_repo_and_is_bucket_are_mutually_exclusive() -> None:
    repo_uri = parse_hf_uri("hf://datasets/org/ds")
    assert repo_uri.is_repo and not repo_uri.is_bucket
    bucket_uri = parse_hf_uri("hf://buckets/org/b")
    assert bucket_uri.is_bucket and not bucket_uri.is_repo


def test_to_string_always_includes_type_prefix_for_models() -> None:
    """Even when the input omitted the ``models/`` prefix, the canonical form keeps it."""
    uri = parse_hf_uri("hf://gpt2")
    assert uri.to_string() == "hf://models/gpt2"


def test_uri_str_is_to_string() -> None:
    uri = parse_hf_uri("hf://datasets/my-org/my-dataset@v1/file.csv")
    assert str(uri) == uri.to_string()
