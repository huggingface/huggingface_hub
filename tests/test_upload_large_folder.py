import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub._upload_large_folder import upload_large_folder_internal
from huggingface_hub.hf_api import HfApi


class TestUploadLargeFolder(unittest.TestCase):
    def setUp(self):
        self.api = Mock(spec=HfApi)
        self.repo_id = "test-repo"
        self.repo_type = "model"
        self.revision = "main"
        self.private = False
        self.allow_patterns = None
        self.ignore_patterns = None
        self.num_workers = 1
        self.print_report = False
        self.print_report_every = 60

    def test_upload_large_folder_with_symlinks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder_path = Path(tmpdir)
            (folder_path / "file1.txt").write_text("content1")
            (folder_path / "file2.txt").write_text("content2")
            os.symlink(folder_path / "file1.txt", folder_path / "symlink1.txt")
            os.symlink(folder_path / "file2.txt", folder_path / "symlink2.txt")

            upload_large_folder_internal(
                api=self.api,
                repo_id=self.repo_id,
                folder_path=folder_path,
                repo_type=self.repo_type,
                revision=self.revision,
                private=self.private,
                allow_patterns=self.allow_patterns,
                ignore_patterns=self.ignore_patterns,
                num_workers=self.num_workers,
                print_report=self.print_report,
                print_report_every=self.print_report_every,
                recurse_symlinks=True,
            )

            self.api.create_repo.assert_called_once_with(repo_id=self.repo_id, repo_type=self.repo_type, private=self.private, exist_ok=True)
            self.api.create_commit.assert_called_once()
            self.assertEqual(len(self.api.create_commit.call_args[1]["operations"]), 4)

    def test_upload_large_folder_without_symlinks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder_path = Path(tmpdir)
            (folder_path / "file1.txt").write_text("content1")
            (folder_path / "file2.txt").write_text("content2")
            os.symlink(folder_path / "file1.txt", folder_path / "symlink1.txt")
            os.symlink(folder_path / "file2.txt", folder_path / "symlink2.txt")

            upload_large_folder_internal(
                api=self.api,
                repo_id=self.repo_id,
                folder_path=folder_path,
                repo_type=self.repo_type,
                revision=self.revision,
                private=self.private,
                allow_patterns=self.allow_patterns,
                ignore_patterns=self.ignore_patterns,
                num_workers=self.num_workers,
                print_report=self.print_report,
                print_report_every=self.print_report_every,
                recurse_symlinks=False,
            )

            self.api.create_repo.assert_called_once_with(repo_id=self.repo_id, repo_type=self.repo_type, private=self.private, exist_ok=True)
            self.api.create_commit.assert_called_once()
            self.assertEqual(len(self.api.create_commit.call_args[1]["operations"]), 2)
