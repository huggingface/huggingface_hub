import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub.commands.delete_cache import DeleteCacheCommand
from huggingface_hub.commands.scan_cache import ScanCacheCommand
from huggingface_hub.commands.upload import UploadCommand
from huggingface_hub.utils import SoftTemporaryDirectory

from .testing_utils import DUMMY_MODEL_ID


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.

        TODO: add other subcommands.
        """
        self.parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        ScanCacheCommand.register_subcommand(commands_parser)
        DeleteCacheCommand.register_subcommand(commands_parser)

    def test_scan_cache_basic(self) -> None:
        """Test `huggingface-cli scan-cache`."""
        args = self.parser.parse_args(["scan-cache"])
        self.assertEqual(args.dir, None)
        self.assertEqual(args.verbose, 0)
        self.assertEqual(args.func, ScanCacheCommand)

    def test_scan_cache_verbose(self) -> None:
        """Test `huggingface-cli scan-cache -v`."""
        args = self.parser.parse_args(["scan-cache", "-v"])
        self.assertEqual(args.dir, None)
        self.assertEqual(args.verbose, 1)
        self.assertEqual(args.func, ScanCacheCommand)

    def test_scan_cache_with_dir(self) -> None:
        """Test `huggingface-cli scan-cache --dir something`."""
        args = self.parser.parse_args(["scan-cache", "--dir", "something"])
        self.assertEqual(args.dir, "something")
        self.assertEqual(args.verbose, 0)
        self.assertEqual(args.func, ScanCacheCommand)

    def test_scan_cache_ultra_verbose(self) -> None:
        """Test `huggingface-cli scan-cache -vvv`."""
        args = self.parser.parse_args(["scan-cache", "-vvv"])
        self.assertEqual(args.dir, None)
        self.assertEqual(args.verbose, 3)
        self.assertEqual(args.func, ScanCacheCommand)

    def test_delete_cache_with_dir(self) -> None:
        """Test `huggingface-cli delete-cache --dir something`."""
        args = self.parser.parse_args(["delete-cache", "--dir", "something"])
        self.assertEqual(args.dir, "something")
        self.assertEqual(args.func, DeleteCacheCommand)


class TestUploadCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.
        """
        self.parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        UploadCommand.register_subcommand(commands_parser)

    def test_upload_basic(self) -> None:
        """Test `huggingface-cli upload my-file to dummy-repo`."""
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "my-file"]))
        self.assertEqual(cmd.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(cmd.local_path, "my-file")
        self.assertEqual(cmd.path_in_repo, "my-file")  # implicit
        self.assertEqual(cmd.repo_type, "model")
        self.assertEqual(cmd.revision, None)
        self.assertEqual(cmd.include, None)
        self.assertEqual(cmd.exclude, None)
        self.assertEqual(cmd.delete, None)
        self.assertEqual(cmd.commit_message, None)
        self.assertEqual(cmd.commit_description, None)
        self.assertEqual(cmd.create_pr, False)
        self.assertEqual(cmd.every, None)
        self.assertEqual(cmd.token, None)
        self.assertEqual(cmd.quiet, False)

    def test_upload_with_all_options(self) -> None:
        """Test `huggingface-cli upload my-file to dummy-repo with all options selected`."""
        cmd = UploadCommand(
            self.parser.parse_args(
                [
                    "upload",
                    DUMMY_MODEL_ID,
                    "my-file",
                    "/",
                    "--repo-type",
                    "dataset",
                    "--revision",
                    "v1.0.0",
                    "--include",
                    "*.json",
                    "*.yaml",
                    "--exclude",
                    "*.log",
                    "*.txt",
                    "--delete",
                    "*.config",
                    "*.secret",
                    "--commit-message",
                    "My commit message",
                    "--commit-description",
                    "My commit description",
                    "--create-pr",
                    "--every",
                    "5",
                    "--token",
                    "my-token",
                    "--quiet",
                ]
            )
        )
        self.assertEqual(cmd.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(cmd.local_path, "my-file")
        self.assertEqual(cmd.path_in_repo, "/")
        self.assertEqual(cmd.repo_type, "dataset")
        self.assertEqual(cmd.revision, "v1.0.0")
        self.assertEqual(cmd.include, ["*.json", "*.yaml"])
        self.assertEqual(cmd.exclude, ["*.log", "*.txt"])
        self.assertEqual(cmd.delete, ["*.config", "*.secret"])
        self.assertEqual(cmd.commit_message, "My commit message")
        self.assertEqual(cmd.commit_description, "My commit description")
        self.assertEqual(cmd.create_pr, True)
        self.assertEqual(cmd.every, 5)
        self.assertEqual(cmd.token, "my-token")
        self.assertEqual(cmd.quiet, True)

    def test_upload_implicit_paths(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo"]))
        self.assertEqual(cmd.local_path, ".")
        self.assertEqual(cmd.path_in_repo, ".")

    def test_upload_explicit_local_path_implicit_path_in_repo(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo", "./path/to/folder"]))
        self.assertEqual(cmd.local_path, "./path/to/folder")
        self.assertEqual(cmd.path_in_repo, "path/to/folder")

    def test_upload_explicit_paths(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo", "./path/to/folder", "data/"]))
        self.assertEqual(cmd.local_path, "./path/to/folder")
        self.assertEqual(cmd.path_in_repo, "data/")

    def test_cannot_upload_verbose_and_quiet(self) -> None:
        with self.assertRaises(ValueError):
            UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "my-file", "--quiet", "--verbose"]))

    def test_every_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "--every", "0"]))

        with self.assertRaises(ValueError):
            UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "--every", "-10"]))

    def test_every_as_int(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "--every", "10"]))
        self.assertEqual(cmd.every, 10)

    def test_every_as_float(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "--every", "0.5"]))
        self.assertEqual(cmd.every, 0.5)

    @patch("huggingface_hub.commands.upload.upload_folder")
    @patch("huggingface_hub.commands.upload.create_repo")
    def test_upload_folder_mock(self, create_mock: Mock, upload_mock: Mock) -> None:
        with SoftTemporaryDirectory() as cache_dir:
            cmd = UploadCommand(
                self.parser.parse_args(
                    ["upload", "my-model", cache_dir, ".", "--private", "--include", "*.json", "--delete", "*.json"]
                )
            )
            cmd.run()

            create_mock.assert_called_once_with(
                repo_id="my-model", repo_type="model", exist_ok=True, private=True, token=None
            )
            upload_mock.assert_called_once_with(
                folder_path=cache_dir,
                path_in_repo=".",
                repo_id=create_mock.return_value.repo_id,
                repo_type="model",
                revision=None,
                token=None,
                commit_message=None,
                commit_description=None,
                create_pr=False,
                allow_patterns=["*.json"],
                ignore_patterns=None,
                delete_patterns=["*.json"],
            )

    @patch("huggingface_hub.commands.upload.upload_file")
    @patch("huggingface_hub.commands.upload.create_repo")
    def test_upload_file_mock(self, create_mock: Mock, upload_mock: Mock) -> None:
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            cmd = UploadCommand(
                self.parser.parse_args(
                    ["upload", "my-dataset", str(file_path), "logs/file.txt", "--repo-type", "dataset", "--create-pr"]
                )
            )
            cmd.run()

            create_mock.assert_called_once_with(
                repo_id="my-dataset", repo_type="dataset", exist_ok=True, private=False, token=None
            )
            upload_mock.assert_called_once_with(
                path_or_fileobj=str(file_path),
                path_in_repo="logs/file.txt",
                repo_id=create_mock.return_value.repo_id,
                repo_type="dataset",
                revision=None,
                token=None,
                commit_message=None,
                commit_description=None,
                create_pr=True,
            )

    @patch("huggingface_hub.commands.upload.create_repo")
    def test_upload_missing_path(self, create_mock: Mock) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-model", "/path/to/missing_file", "logs/file.txt"]))
        with self.assertRaises(FileNotFoundError):
            cmd.run()  # File/folder does not exist locally

        # Repo creation happens before the check
        create_mock.assert_not_called()
