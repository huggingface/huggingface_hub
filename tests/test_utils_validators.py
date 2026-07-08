from pathlib import Path
from unittest.mock import Mock

import pytest

from huggingface_hub.utils import (
    HFValidationError,
    validate_hf_hub_args,
    validate_repo_id,
)


class TestHfHubValidator:
    """Test `validate_hf_hub_args` decorator calls all default validators."""

    def test_validate_repo_id_as_arg(self, mocker) -> None:
        """Test `validate_repo_id` is called when `repo_id` is passed as arg."""
        validate_repo_id_mock: Mock = mocker.patch("huggingface_hub.utils._validators.validate_repo_id")
        self.dummy_function(123)
        validate_repo_id_mock.assert_called_once_with(123)

    def test_validate_repo_id_as_kwarg(self, mocker) -> None:
        """Test `validate_repo_id` is called when `repo_id` is passed as kwarg."""
        validate_repo_id_mock: Mock = mocker.patch("huggingface_hub.utils._validators.validate_repo_id")
        self.dummy_function(repo_id=123)
        validate_repo_id_mock.assert_called_once_with(123)

    @staticmethod
    @validate_hf_hub_args
    def dummy_function(repo_id: str) -> None:
        pass


class TestRepoIdValidator:
    VALID_VALUES = (
        "123",
        "foo",
        "foo/bar",
        "Foo-BAR_foo.bar123",
        None,
    )
    NOT_VALID_VALUES = (
        Path("foo/bar"),  # Must be a string
        "a" * 100,  # Too long
        "datasets/foo/bar",  # Repo_type forbidden in repo_id
        ".repo_id",  # Cannot start with .
        "repo_id.",  # Cannot end with .
        "foo--bar",  # Cannot contain "--"
        "foo..bar",  # Cannot contain "."
        "foo.git",  # Cannot end with ".git"
    )

    def test_valid_repo_ids(self) -> None:
        """Test `repo_id` validation on valid values."""
        for repo_id in self.VALID_VALUES:
            validate_repo_id(repo_id)

    def test_not_valid_repo_ids(self) -> None:
        """Test `repo_id` validation on not valid values."""
        for repo_id in self.NOT_VALID_VALUES:
            with pytest.raises(HFValidationError):
                validate_repo_id(repo_id)
