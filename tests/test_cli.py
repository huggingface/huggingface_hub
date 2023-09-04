import unittest
import warnings
from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch

from huggingface_hub.commands.delete_cache import DeleteCacheCommand
from huggingface_hub.commands.download import DownloadCommand
from huggingface_hub.commands.scan_cache import ScanCacheCommand
from huggingface_hub.utils import capture_output

from .testing_utils import (
    DUMMY_MODEL_ID,
)


class TestCacheCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up scan-cache/delete-cache commands as in `src/huggingface_hub/commands/huggingface_cli.py`.
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

    @patch("huggingface_hub.commands.download.hf_hub_download")
    def test_download_file_from_revision(self, mock: Mock) -> None:
        args = Namespace(
            token="hf_****",
            repo_id="author/dataset",
            filenames=["README.md"],
            repo_type="dataset",
            revision="refs/pr/1",
            include=None,
            exclude=None,
            force_download=False,
            resume_download=False,
            cache_dir=None,
            quiet=False,
        )

        # Output path is printed to terminal once run is completed
        with capture_output() as output:
            DownloadCommand(args).run()
        self.assertRegex(output.getvalue(), r"<MagicMock name='hf_hub_download\(\)' id='\d+'>")

        mock.assert_called_once_with(
            repo_id="author/dataset",
            repo_type="dataset",
            revision="refs/pr/1",
            filename="README.md",
            cache_dir=None,
            resume_download=False,
            force_download=False,
            token="hf_****",
        )

    @patch("huggingface_hub.commands.download.snapshot_download")
    def test_download_multiple_files(self, mock: Mock) -> None:
        args = Namespace(
            token="hf_****",
            repo_id="author/model",
            filenames=["README.md", "config.json"],
            repo_type="model",
            revision=None,
            include=None,
            exclude=None,
            force_download=True,
            resume_download=True,
            cache_dir=None,
            quiet=False,
        )
        DownloadCommand(args).run()

        # Use `snapshot_download` to ensure all files comes from same revision
        mock.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["README.md", "config.json"],
            ignore_patterns=None,
            resume_download=True,
            force_download=True,
            cache_dir=None,
            token="hf_****",
        )

    @patch("huggingface_hub.commands.download.snapshot_download")
    def test_download_with_patterns(self, mock: Mock) -> None:
        args = Namespace(
            token=None,
            repo_id="author/model",
            filenames=[],
            repo_type="model",
            revision=None,
            include=["*.json"],
            exclude=["data/*"],
            force_download=True,
            resume_download=True,
            cache_dir=None,
            quiet=False,
        )
        DownloadCommand(args).run()

        # Use `snapshot_download` to ensure all files comes from same revision
        mock.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["*.json"],
            ignore_patterns=["data/*"],
            resume_download=True,
            force_download=True,
            cache_dir=None,
            token=None,
        )

    @patch("huggingface_hub.commands.download.snapshot_download")
    def test_download_with_ignored_patterns(self, mock: Mock) -> None:
        args = Namespace(
            token=None,
            repo_id="author/model",
            filenames=["README.md", "config.json"],
            repo_type="model",
            revision=None,
            include=["*.json"],
            exclude=["data/*"],
            force_download=True,
            resume_download=True,
            cache_dir=None,
            quiet=False,
        )

        with self.assertWarns(UserWarning):
            # warns that patterns are ignored
            DownloadCommand(args).run()

        mock.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["README.md", "config.json"],  # `filenames` has priority over the patterns
            ignore_patterns=None,  # cleaned up
            resume_download=True,
            force_download=True,
            cache_dir=None,
            token=None,
        )

        # Same but quiet (no warnings)
        args.quiet = True
        with warnings.catch_warnings():
            # Taken from https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
            warnings.simplefilter("error")
            DownloadCommand(args).run()
