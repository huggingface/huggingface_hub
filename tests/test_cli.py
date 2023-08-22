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
        command = DownloadCommand("")
        command.register_subcommand(commands_parser)

    def test_download_basic(self) -> None:
        """Test `huggingface-cli download my-repo`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, False)
        print("TEST!!!", args.func)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_type(self) -> None:
        """Test `huggingface-cli download my-repo --type model`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--type", "model"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, "model")
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_revision(self) -> None:
        """Test `huggingface-cli download my-repo --revision v1.0.0`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--revision", "v1.0.0"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, "v1.0.0")
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_allow_patterns(self) -> None:
        """Test `huggingface-cli download my-repo --allow-patterns "*.json" "*.yaml"`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--allow-patterns", "*.json", "*.yaml"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, ["*.json", "*.yaml"])
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_ignore_patterns(self) -> None:
        """Test `huggingface-cli download my-repo --ignore-patterns "*.log" "*.txt"`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--ignore-patterns", "*.log", "*.txt"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, ["*.log", "*.txt"])
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_to_local_dir(self) -> None:
        """Test `huggingface-cli download my-repo --to-local-dir /tmp/my-repo`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--to-local-dir", "/tmp/my-repo"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, "/tmp/my-repo")
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_local_dir_use_symlinks(self) -> None:
        """Test `huggingface-cli download my-repo --local-dir-use-symlinks`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--local-dir-use-symlinks"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, True)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_proxies(self) -> None:
        """Test `huggingface-cli download my-repo --proxies http://127.0.0.1:8080`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--proxies", "http://127.0.0.1:8080"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, {"http": "http://127.0.0.1:8080"})
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_force_download(self) -> None:
        """Test `huggingface-cli download my-repo --force-download`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID, "--force-download"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, None)
        self.assertEqual(args.revision, None)
        self.assertEqual(args.allow_patterns, None)
        self.assertEqual(args.ignore_patterns, None)
        self.assertEqual(args.to_local_dir, None)
        self.assertEqual(args.local_dir_use_symlinks, False)
        self.assertEqual(args.proxies, None)
        self.assertEqual(args.force_download, True)
        self.assertEqual(args.func, DownloadCommand)

    def test_download_with_all_options(self) -> None:
        """Test `huggingface-cli download my-repo --type model --revision v1.0.0 --allow-patterns "*.json" "*.yaml" --ignore-patterns "*.log" "*.txt" --to-local-dir /tmp/my-repo --local-dir-use-symlinks --proxies http://127.0.0.1:8080 --force-download`."""
        args = self.parser.parse_args(
            [
                "download",
                DUMMY_MODEL_ID,
                "--type",
                "model",
                "--revision",
                "v1.0.0",
                "--allow-patterns",
                "*.json",
                "*.yaml",
                "--ignore-patterns",
                "*.log",
                "*.txt",
                "--to-local-dir",
                "/tmp/my-repo",
                "--local-dir-use-symlinks",
                "--proxies",
                "http://127.0.0.1:8080",
                "--force-download",
            ]
        )
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.type, "model")
        self.assertEqual(args.revision, "v1.0.0")
        self.assertEqual(args.allow_patterns, ["*.json", "*.yaml"])
        self.assertEqual(args.ignore_patterns, ["*.log", "*.txt"])
        self.assertEqual(args.to_local_dir, "/tmp/my-repo")
        self.assertEqual(args.local_dir_use_symlinks, True)
        self.assertEqual(args.proxies, {"http": "http://127.0.0.1:8080"})
        self.assertEqual(args.force_download, True)
        self.assertEqual(args.func, DownloadCommand)
