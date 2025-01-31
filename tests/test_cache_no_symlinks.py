import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from huggingface_hub import hf_hub_download, scan_cache_dir
from huggingface_hub.constants import CONFIG_NAME, HF_HUB_CACHE
from huggingface_hub.file_download import are_symlinks_supported

from .testing_utils import DUMMY_MODEL_ID, with_production_testing


@with_production_testing
@pytest.mark.usefixtures("fx_cache_dir")
class TestCacheLayoutIfSymlinksNotSupported(unittest.TestCase):
    cache_dir: Path

    @patch(
        "huggingface_hub.file_download._are_symlinks_supported_in_dir",
        {HF_HUB_CACHE: True},
    )
    def test_are_symlinks_supported_default(self) -> None:
        self.assertTrue(are_symlinks_supported())

    @patch("huggingface_hub.file_download.os.symlink")
    @patch("huggingface_hub.file_download._are_symlinks_supported_in_dir", {})
    def test_are_symlinks_supported_windows_specific_dir(self, mock_symlink: Mock) -> None:
        mock_symlink.side_effect = [OSError(), None]  # First dir not supported then yes
        this_dir = Path(__file__).parent

        # First time in `this_dir`: warning is raised
        with self.assertWarns(UserWarning):
            self.assertFalse(are_symlinks_supported(this_dir))

        with warnings.catch_warnings():
            # Assert no warnings raised
            # Taken from https://stackoverflow.com/a/45671804
            warnings.simplefilter("error")

            # Second time in `this_dir` but with absolute path: value is still cached
            self.assertFalse(are_symlinks_supported(this_dir.absolute()))

            # Try with another directory: symlinks are supported, no warnings
            self.assertTrue(are_symlinks_supported())  # True

    @patch("huggingface_hub.file_download.are_symlinks_supported")
    def test_download_no_symlink_new_file(self, mock_are_symlinks_supported: Mock) -> None:
        mock_are_symlinks_supported.return_value = False
        filepath = Path(
            hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                cache_dir=self.cache_dir,
                local_files_only=False,
            )
        )
        # Not a symlink !
        self.assertFalse(filepath.is_symlink())
        self.assertTrue(filepath.is_file())

        # Blobs directory is empty
        self.assertEqual(len(list((Path(filepath).parents[2] / "blobs").glob("*"))), 0)

    @patch("huggingface_hub.file_download.are_symlinks_supported")
    def test_download_no_symlink_existing_file(self, mock_are_symlinks_supported: Mock) -> None:
        mock_are_symlinks_supported.return_value = True
        filepath = Path(
            hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                cache_dir=self.cache_dir,
                local_files_only=False,
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
            )
        )
        # File exist but is not a symlink
        self.assertFalse(new_filepath.is_symlink())
        self.assertTrue(new_filepath.is_file())

        # Blob file still exists as well (has not been deleted)
        # => duplicate file on disk
        self.assertTrue(blob_path.is_file())

    @patch("huggingface_hub.file_download.are_symlinks_supported")
    def test_scan_and_delete_cache_no_symlinks(self, mock_are_symlinks_supported: Mock) -> None:
        """Test scan_cache_dir works as well when cache-system doesn't use symlinks."""
        OLDER_REVISION = "44c70f043cfe8162efc274ff531575e224a0e6f0"

        # Symlinks not supported
        mock_are_symlinks_supported.return_value = False

        # Download config.json from main
        hf_hub_download(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            cache_dir=self.cache_dir,
        )

        # Download README.md from main
        hf_hub_download(
            DUMMY_MODEL_ID,
            filename="README.md",
            cache_dir=self.cache_dir,
        )

        # Download config.json from older revision
        hf_hub_download(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            cache_dir=self.cache_dir,
            revision=OLDER_REVISION,
        )

        # Now symlinks work: user has rerun the script as admin
        mock_are_symlinks_supported.return_value = True

        # Download merges.txt from older revision with symlinks
        hf_hub_download(
            DUMMY_MODEL_ID,
            filename="merges.txt",
            cache_dir=self.cache_dir,
            revision=OLDER_REVISION,
        )

        # Scan cache directory
        report = scan_cache_dir(self.cache_dir)

        # 1 repo found, no warnings
        self.assertEqual(len(report.repos), 1)
        self.assertEqual(len(report.warnings), 0)
        repo = list(report.repos)[0]

        # 2 revisions found
        self.assertEqual(len(repo.revisions), 2)
        self.assertEqual(repo.nb_files, 4)
        self.assertEqual(len(repo.refs), 1)  # only `main`
        main_revision = repo.refs["main"]
        main_ref = main_revision.commit_hash
        older_revision = [rev for rev in repo.revisions if rev is not main_revision][0]

        # 2 files in `main` revisions, both are not symlinks
        self.assertEqual(main_revision.nb_files, 2)
        for file in main_revision.files:
            # No symlinks means the files are in the snapshot dir itself
            self.assertTrue(main_revision.snapshot_path in file.blob_path.parents)

        # 2 files in older revision, only 1 as symlink
        for file in older_revision.files:
            if file.file_name == CONFIG_NAME:
                # In snapshot dir
                self.assertTrue(older_revision.snapshot_path in file.blob_path.parents)
            else:
                # In blob dir
                self.assertFalse(older_revision.snapshot_path in file.blob_path.parents)
                self.assertTrue("blobs" in str(file.blob_path))

        # Since files are not shared (README.md is duplicated in cache), the total size
        # of the repo is the sum of each revision size. If symlinks were used, the total
        # size of the repo would be lower.
        self.assertEqual(repo.size_on_disk, main_revision.size_on_disk + older_revision.size_on_disk)

        # Test delete repo strategy
        strategy_delete_repo = report.delete_revisions(main_ref, OLDER_REVISION)
        self.assertEqual(strategy_delete_repo.expected_freed_size, repo.size_on_disk)
        self.assertEqual(len(strategy_delete_repo.blobs), 0)
        self.assertEqual(len(strategy_delete_repo.snapshots), 0)
        self.assertEqual(len(strategy_delete_repo.refs), 0)
        self.assertEqual(len(strategy_delete_repo.repos), 1)

        # Test delete older revision strategy
        strategy_delete_revision = report.delete_revisions(OLDER_REVISION)
        self.assertEqual(
            strategy_delete_revision.blobs,
            {file.blob_path for file in older_revision.files},
        )
        self.assertEqual(strategy_delete_revision.snapshots, {older_revision.snapshot_path})
        self.assertEqual(len(strategy_delete_revision.refs), 0)
        self.assertEqual(len(strategy_delete_revision.repos), 0)
        strategy_delete_revision.execute()  # Execute without error
