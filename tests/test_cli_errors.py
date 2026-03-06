"""Tests for CLI error formatting utilities."""

from unittest.mock import Mock

import httpx

from huggingface_hub.cli._errors import (
    _format_bucket_not_found,
    _format_entry_not_found,
    _format_gated_repo,
    _format_repo_not_found,
    _format_revision_not_found,
)
from huggingface_hub.errors import (
    BucketNotFoundError,
    GatedRepoError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


def _make_error(cls, **attrs):
    """Helper to create an HfHubHTTPError subclass with custom attributes."""
    response = Mock(spec=httpx.Response)
    response.headers = httpx.Headers({})
    response.request = Mock(spec=httpx.Request)
    err = cls("test", response=response)
    for key, value in attrs.items():
        setattr(err, key, value)
    return err


class TestFormatRepoNotFound:
    def test_with_repo_id_and_type(self):
        err = _make_error(RepositoryNotFoundError, repo_id="user/repo", repo_type="model")
        assert (
            _format_repo_not_found(err)
            == "Model 'user/repo' not found. If the repo is private, make sure you are authenticated."
        )

    def test_with_repo_id_dataset(self):
        err = _make_error(RepositoryNotFoundError, repo_id="user/data", repo_type="dataset")
        assert "Dataset 'user/data' not found." in _format_repo_not_found(err)

    def test_with_repo_id_no_type(self):
        err = _make_error(RepositoryNotFoundError, repo_id="user/repo", repo_type=None)
        assert "Repository 'user/repo' not found." in _format_repo_not_found(err)

    def test_without_repo_id(self):
        err = _make_error(RepositoryNotFoundError, repo_id=None, repo_type=None)
        msg = _format_repo_not_found(err)
        assert "Repository not found." in msg
        assert "authenticated" in msg


class TestFormatGatedRepo:
    def test_with_repo_id(self):
        err = _make_error(GatedRepoError, repo_id="user/gated")
        assert _format_gated_repo(err) == "Access denied. Repository 'user/gated' requires approval."

    def test_without_repo_id(self):
        err = _make_error(GatedRepoError, repo_id=None)
        assert _format_gated_repo(err) == "Access denied. This repository requires approval."


class TestFormatBucketNotFound:
    def test_with_bucket_id(self):
        err = _make_error(BucketNotFoundError, bucket_id="ns/bucket")
        msg = _format_bucket_not_found(err)
        assert "Bucket 'ns/bucket' not found." in msg
        assert "authenticated" in msg

    def test_without_bucket_id(self):
        err = _make_error(BucketNotFoundError, bucket_id=None)
        msg = _format_bucket_not_found(err)
        assert "Bucket not found." in msg
        assert "namespace/name" in msg


class TestFormatEntryNotFound:
    def test_with_repo_id_and_type(self):
        err = _make_error(RemoteEntryNotFoundError, repo_id="user/repo", repo_type="dataset")
        msg = _format_entry_not_found(err)
        assert "File not found in dataset 'user/repo'." in msg

    def test_with_repo_id_no_type(self):
        err = _make_error(RemoteEntryNotFoundError, repo_id="user/repo", repo_type=None)
        msg = _format_entry_not_found(err)
        assert "File not found in repository 'user/repo'." in msg

    def test_without_repo_id(self):
        err = _make_error(RemoteEntryNotFoundError, repo_id=None, repo_type=None)
        msg = _format_entry_not_found(err)
        assert "File not found in repository." in msg

    def test_includes_url(self):
        err = _make_error(RemoteEntryNotFoundError, repo_id="user/repo", repo_type="model")
        err.response.url = "https://huggingface.co/api/models/user/repo/resolve/main/missing.bin"
        msg = _format_entry_not_found(err)
        assert "File not found in model 'user/repo'." in msg
        assert "URL: https://huggingface.co/api/models/user/repo/resolve/main/missing.bin" in msg


class TestFormatRevisionNotFound:
    def test_with_repo_id(self):
        err = _make_error(RevisionNotFoundError, repo_id="user/repo", repo_type=None)
        assert _format_revision_not_found(err) == "Revision not found in repository 'user/repo'."

    def test_with_repo_id_and_type(self):
        err = _make_error(RevisionNotFoundError, repo_id="user/repo", repo_type="dataset")
        assert _format_revision_not_found(err) == "Revision not found in dataset 'user/repo'."

    def test_without_repo_id(self):
        err = _make_error(RevisionNotFoundError, repo_id=None, repo_type=None)
        assert _format_revision_not_found(err) == "Revision not found in repository. Check the revision parameter."
