"""Tests for huggingface_hub.utils._hf_uri – centralised HF path parsing."""

import pytest

from huggingface_hub.utils._hf_uri import (
    ParsedBucketUrl,
    ParsedHfUrl,
    is_bucket_url,
    is_hf_url,
    parse_hf_url,
)


@pytest.mark.parametrize(
    "url, expected",
    [
        # bare / namespaced repo ids (no explicit type → has_explicit_type=False)
        ("gpt2", ParsedHfUrl("model", "gpt2", None, "")),
        ("username/my-model", ParsedHfUrl("model", "username/my-model", None, "")),
        # hf:// prefix
        ("hf://gpt2", ParsedHfUrl("model", "gpt2", None, "")),
        ("hf://username/my-model", ParsedHfUrl("model", "username/my-model", None, "")),
        # explicit type prefixes (has_explicit_type=True)
        ("hf://datasets/squad", ParsedHfUrl("dataset", "squad", None, "", True)),
        ("hf://datasets/username/ds", ParsedHfUrl("dataset", "username/ds", None, "", True)),
        ("hf://spaces/username/app", ParsedHfUrl("space", "username/app", None, "", True)),
        ("hf://models/username/m", ParsedHfUrl("model", "username/m", None, "", True)),
        ("datasets/username/ds", ParsedHfUrl("dataset", "username/ds", None, "", True)),
        # path_in_repo
        ("hf://datasets/user/ds/train/data.csv", ParsedHfUrl("dataset", "user/ds", None, "train/data.csv", True)),
        ("hf://user/model/config.json", ParsedHfUrl("model", "user/model", None, "config.json")),
        # @revision
        ("hf://datasets/user/ds@dev/train.csv", ParsedHfUrl("dataset", "user/ds", "dev", "train.csv", True)),
        ("hf://user/model@v1.0", ParsedHfUrl("model", "user/model", "v1.0", "")),
        ("gpt2@main", ParsedHfUrl("model", "gpt2", "main", "")),
        # special refs revisions
        ("hf://user/m@refs/pr/10/file.txt", ParsedHfUrl("model", "user/m", "refs/pr/10", "file.txt")),
        ("hf://user/m@refs/pr/10", ParsedHfUrl("model", "user/m", "refs/pr/10", "")),
        ("hf://datasets/u/ds@refs/convert/parquet", ParsedHfUrl("dataset", "u/ds", "refs/convert/parquet", "", True)),
        ("hf://datasets/u/ds@refs/convert/parquet/default/train/0000.parquet", ParsedHfUrl("dataset", "u/ds", "refs/convert/parquet", "default/train/0000.parquet", True)),
        # URL-encoded revision
        ("hf://user/m@refs%2Fpr%2F10", ParsedHfUrl("model", "user/m", "refs/pr/10", "")),
        # buckets
        ("hf://buckets/ns/bucket", ParsedBucketUrl("ns/bucket", "")),
        ("hf://buckets/ns/bucket/some/prefix", ParsedBucketUrl("ns/bucket", "some/prefix")),
        ("buckets/ns/bucket", ParsedBucketUrl("ns/bucket", "")),
        ("buckets/ns/bucket/data/logs", ParsedBucketUrl("ns/bucket", "data/logs")),
    ],
)
def test_parse_hf_url(url, expected):
    assert parse_hf_url(url) == expected


def test_default_type_override():
    assert parse_hf_url("user/thing", default_type="dataset").repo_type == "dataset"
    # explicit type prefix wins over default_type
    assert parse_hf_url("spaces/user/app", default_type="dataset").repo_type == "space"


@pytest.mark.parametrize("url", ["", "hf://", "hf://buckets/only-ns", "hf://buckets/"])
def test_parse_hf_url_errors(url):
    with pytest.raises(ValueError):
        parse_hf_url(url)


def test_properties():
    repo = parse_hf_url("hf://datasets/username/ds")
    assert repo.namespace == "username"
    assert repo.repo_name == "ds"

    bare = parse_hf_url("gpt2")
    assert bare.namespace is None
    assert bare.repo_name == "gpt2"

    bucket = parse_hf_url("hf://buckets/org/my-bucket")
    assert bucket.namespace == "org"
    assert bucket.bucket_name == "my-bucket"


def test_is_hf_url():
    assert is_hf_url("hf://datasets/squad") is True
    assert is_hf_url("datasets/squad") is False


def test_is_bucket_url():
    assert is_bucket_url("hf://buckets/ns/name") is True
    assert is_bucket_url("buckets/ns/name") is True
    assert is_bucket_url("hf://datasets/squad") is False
