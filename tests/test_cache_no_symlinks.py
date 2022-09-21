import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.file_download import are_symlinks_supported
from huggingface_hub.utils import logging

from .testing_constants import TOKEN
from .testing_utils import DUMMY_MODEL_ID, with_production_testing


logger = logging.get_logger(__name__)
MODEL_IDENTIFIER = "hf-internal-testing/hfh-cache-layout"


def get_file_contents(path):
    with open(path) as f:
        content = f.read()

    return content


@with_production_testing
@pytest.mark.usefixtures("fx_cache_dir")
class TestCacheLayoutIfSymlinksNotSupported(unittest.TestCase):
    cache_dir: Path

    @patch("huggingface_hub.file_download._are_symlinks_supported", None)
    def test_are_symlinks_supported_normal(self) -> None:
        self.assertTrue(are_symlinks_supported())

    @patch("huggingface_hub.file_download.os.symlink")  # Symlinks not supported
    @patch("huggingface_hub.file_download._are_symlinks_supported", None)  # first use
    def test_are_symlinks_supported_windows(self, mock_symlink: Mock) -> None:
        mock_symlink.side_effect = OSError()

        # First time: warning is raised
        with self.assertWarns(UserWarning):
            self.assertFalse(are_symlinks_supported())

        # Afterward: value is cached
        with warnings.catch_warnings():
            # Taken from https://stackoverflow.com/a/45671804
            warnings.simplefilter("error")
            self.assertFalse(are_symlinks_supported())

    @patch("huggingface_hub.file_download.are_symlinks_supported")
    def test_download_no_symlink_new_file(
        self, mock_are_symlinks_supported: Mock
    ) -> None:
        mock_are_symlinks_supported.return_value = False
        filepath = Path(
            hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                cache_dir=self.cache_dir,
                local_files_only=False,
                use_auth_token=TOKEN,
            )
        )
        # Not a symlink !
        self.assertFalse(filepath.is_symlink())
        self.assertTrue(filepath.is_file())

        # Blobs directory is empty
        self.assertEqual(len(list((Path(filepath).parents[2] / "blobs").glob("*"))), 0)

    @patch("huggingface_hub.file_download.are_symlinks_supported")
    def test_download_no_symlink_existing_file(
        self, mock_are_symlinks_supported: Mock
    ) -> None:
        mock_are_symlinks_supported.return_value = True
        filepath = Path(
            hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                cache_dir=self.cache_dir,
                local_files_only=False,
                use_auth_token=TOKEN,
            )
        )
        self.assertTrue(filepath.is_symlink())
        blob_path = filepath.resolve()
        self.assertTrue(blob_path.is_file())

        # Delete file in snapshot
        filepath.unlink()

        # Re-download but symlinks are not supported anymore (example: not an admin)
        mock_are_symlinks_supported.return_value = False
        new_filepath = Path(
            hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                cache_dir=self.cache_dir,
                local_files_only=False,
                use_auth_token=TOKEN,
            )
        )
        # File exist but is not a symlink
        self.assertFalse(new_filepath.is_symlink())
        self.assertTrue(new_filepath.is_file())

        # Blob file still exists as well (has not been deleted)
        # => duplicate file on disk
        self.assertTrue(blob_path.is_file())
