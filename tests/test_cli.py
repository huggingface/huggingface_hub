import unittest
from argparse import ArgumentParser

from huggingface_hub.commands.delete_cache import DeleteCacheCommand
from huggingface_hub.commands.scan_cache import ScanCacheCommand
from huggingface_hub.commands.upload import UploadCommand

from .testing_utils import (
    DUMMY_MODEL_ID,
)


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
        args = self.parser.parse_args(["upload", DUMMY_MODEL_ID, "my-file"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.path, "my-file")
        self.assertEqual(args.path_in_repo, None)
        self.assertEqual(args.repo_type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.include, None)
        self.assertEqual(args.exclude, None)
        self.assertEqual(args.delete, None)
        self.assertEqual(args.commit_message, None)
        self.assertEqual(args.commit_description, None)
        self.assertEqual(args.create_pr, False)
        self.assertEqual(args.every, False)
        self.assertEqual(args.token, None)
        self.assertEqual(args.quiet, False)
        self.assertEqual(args.func, UploadCommand)

    def test_upload_with_all_options(self) -> None:
        """Test `huggingface-cli upload my-file to dummy-repo with all options selected`."""
        args = self.parser.parse_args(
            [
                "upload",
                DUMMY_MODEL_ID,
                "my-file",
                "/",
                "--repo-type",
                "model",
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
                "--token",
                "my-token",
                "--quiet",
            ]
        )
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.path, "my-file")
        self.assertEqual(args.path_in_repo, "/")
        self.assertEqual(args.repo_type, "model")
        self.assertEqual(args.revision, "v1.0.0")
        self.assertEqual(args.include, ["*.json", "*.yaml"])
        self.assertEqual(args.exclude, ["*.log", "*.txt"])
        self.assertEqual(args.delete, ["*.config", "*.secret"])
        self.assertEqual(args.commit_message, "My commit message")
        self.assertEqual(args.commit_description, "My commit description")
        self.assertEqual(args.create_pr, True)
        self.assertEqual(args.every, True)
        self.assertEqual(args.token, "my-token")
        self.assertEqual(args.quiet, True)
        self.assertEqual(args.func, UploadCommand)
