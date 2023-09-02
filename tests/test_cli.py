import unittest
from argparse import ArgumentParser

from huggingface_hub.commands.delete_cache import DeleteCacheCommand
from huggingface_hub.commands.download import DownloadCommand
from huggingface_hub.commands.scan_cache import ScanCacheCommand

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


class TestDownloadCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.
        """
        self.parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        DownloadCommand.register_subcommand(commands_parser)

    def test_download_basic(self) -> None:
        """Test `huggingface-cli download dummy-repo`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(len(args.filenames), 0)
        self.assertEqual(args.repo_type, "model")
        self.assertEqual(args.revision, None)
        self.assertEqual(args.include, None)
        self.assertEqual(args.exclude, None)
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.cache_dir, None)
        self.assertEqual(args.resume_download, False)
        self.assertEqual(args.token, None)
        self.assertEqual(args.quiet, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_all_options(self) -> None:
        """Test `huggingface-cli download dummy-repo` with all options selected."""
        args = self.parser.parse_args(
            [
                "download",
                DUMMY_MODEL_ID,
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
                "--force-download",
                "--cache-dir",
                "/tmp",
                "--resume-download",
                "--token",
                "my-token",
                "--quiet",
            ]
        )
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.repo_type, "dataset")
        self.assertEqual(args.revision, "v1.0.0")
        self.assertEqual(args.include, ["*.json", "*.yaml"])
        self.assertEqual(args.exclude, ["*.log", "*.txt"])
        self.assertEqual(args.force_download, True)
        self.assertEqual(args.cache_dir, "/tmp")
        self.assertEqual(args.resume_download, True)
        self.assertEqual(args.token, "my-token")
        self.assertEqual(args.quiet, True)
        self.assertEqual(args.func, DownloadCommand)
