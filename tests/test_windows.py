"""Contains tests that are specific to windows machines."""

import os

import pytest

from huggingface_hub.file_download import are_symlinks_supported


@pytest.mark.skipif(os.name != "nt", reason="test of git lfs workflow")
class TestWindows:
    def test_are_symlink_supported(self) -> None:
        assert not are_symlinks_supported()
