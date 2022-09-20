import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
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
@patch("huggingface_hub.file_download.are_symlinks_supported")
class TestCacheLayoutIfSymlinksNotSupported(unittest.TestCase):
    cache_dir: Path

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
