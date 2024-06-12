import os
import unittest
import warnings
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

from huggingface_hub.commands.delete_cache import DeleteCacheCommand
from huggingface_hub.commands.download import DownloadCommand
from huggingface_hub.commands.repo_files import DeleteFilesSubCommand, RepoFilesCommand
from huggingface_hub.commands.scan_cache import ScanCacheCommand
from huggingface_hub.commands.tag import TagCommands
from huggingface_hub.commands.upload import UploadCommand
from huggingface_hub.utils import RevisionNotFoundError, SoftTemporaryDirectory, capture_output

from .testing_utils import DUMMY_MODEL_ID


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


class TestUploadCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.
        """
        self.parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        UploadCommand.register_subcommand(commands_parser)

    def test_upload_basic(self) -> None:
        """Test `huggingface-cli upload my-folder to dummy-repo`."""
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "my-folder"]))
        self.assertEqual(cmd.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(cmd.local_path, "my-folder")
        self.assertEqual(cmd.path_in_repo, ".")  # implicit
        self.assertEqual(cmd.repo_type, "model")
        self.assertEqual(cmd.revision, None)
        self.assertEqual(cmd.include, None)
        self.assertEqual(cmd.exclude, None)
        self.assertEqual(cmd.delete, None)
        self.assertEqual(cmd.commit_message, None)
        self.assertEqual(cmd.commit_description, None)
        self.assertEqual(cmd.create_pr, False)
        self.assertEqual(cmd.every, None)
        self.assertEqual(cmd.api.token, None)
        self.assertEqual(cmd.quiet, False)

    def test_upload_with_all_options(self) -> None:
        """Test `huggingface-cli upload my-file to dummy-repo with all options selected`."""
        cmd = UploadCommand(
            self.parser.parse_args(
                [
                    "upload",
                    DUMMY_MODEL_ID,
                    "my-file",
                    "data/",
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
        self.assertEqual(cmd.path_in_repo, "data/")
        self.assertEqual(cmd.repo_type, "dataset")
        self.assertEqual(cmd.revision, "v1.0.0")
        self.assertEqual(cmd.include, ["*.json", "*.yaml"])
        self.assertEqual(cmd.exclude, ["*.log", "*.txt"])
        self.assertEqual(cmd.delete, ["*.config", "*.secret"])
        self.assertEqual(cmd.commit_message, "My commit message")
        self.assertEqual(cmd.commit_description, "My commit description")
        self.assertEqual(cmd.create_pr, True)
        self.assertEqual(cmd.every, 5)
        self.assertEqual(cmd.api.token, "my-token")
        self.assertEqual(cmd.quiet, True)

    def test_upload_implicit_local_path_when_folder_exists(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.mkdir()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-cool-model"]))

        # A folder with the same name as the repo exists => upload it at the root of the repo
        self.assertEqual(cmd.local_path, "my-cool-model")
        self.assertEqual(cmd.path_in_repo, ".")

    def test_upload_implicit_local_path_when_file_exists(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.touch()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-cool-model"]))

        # A file with the same name as the repo exists => upload it at the root of the repo
        self.assertEqual(cmd.local_path, "my-cool-model")
        self.assertEqual(cmd.path_in_repo, "my-cool-model")

    def test_upload_implicit_local_path_when_org_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.mkdir()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-cool-org/my-cool-model"]))

        # A folder with the same name as the repo exists => upload it at the root of the repo
        self.assertEqual(cmd.local_path, "my-cool-model")
        self.assertEqual(cmd.path_in_repo, ".")

    def test_upload_implicit_local_path_otherwise(self) -> None:
        # No folder or file has the same name as the repo => raise exception
        with self.assertRaises(ValueError):
            with tmp_current_directory():
                UploadCommand(self.parser.parse_args(["upload", "my-cool-model"]))

    def test_upload_explicit_local_path_to_folder_implicit_path_in_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "path" / "to" / "folder"
            folder_path.mkdir(parents=True, exist_ok=True)
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo", "./path/to/folder"]))
        self.assertEqual(cmd.local_path, "./path/to/folder")
        self.assertEqual(cmd.path_in_repo, ".")  # Always upload the folder at the root of the repo

    def test_upload_explicit_local_path_to_file_implicit_path_in_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            file_path = Path(cache_dir) / "path" / "to" / "file.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo", "./path/to/file.txt"]))
        self.assertEqual(cmd.local_path, "./path/to/file.txt")
        self.assertEqual(cmd.path_in_repo, "file.txt")  # If a file, upload it at the root of the repo and keep name

    def test_upload_explicit_paths(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo", "./path/to/folder", "data/"]))
        self.assertEqual(cmd.local_path, "./path/to/folder")
        self.assertEqual(cmd.path_in_repo, "data/")

    def test_every_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "0"]))

        with self.assertRaises(ValueError):
            UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "-10"]))

    def test_every_as_int(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "10"]))
        self.assertEqual(cmd.every, 10)

    def test_every_as_float(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "0.5"]))
        self.assertEqual(cmd.every, 0.5)

    @patch("huggingface_hub.commands.upload.HfApi.repo_info")
    @patch("huggingface_hub.commands.upload.HfApi.upload_folder")
    @patch("huggingface_hub.commands.upload.HfApi.create_repo")
    def test_upload_folder_mock(self, create_mock: Mock, upload_mock: Mock, repo_info_mock: Mock) -> None:
        with SoftTemporaryDirectory() as cache_dir:
            cache_path = cache_dir.absolute().as_posix()
            cmd = UploadCommand(
                self.parser.parse_args(
                    ["upload", "my-model", cache_path, ".", "--private", "--include", "*.json", "--delete", "*.json"]
                )
            )
            cmd.run()

            create_mock.assert_called_once_with(
                repo_id="my-model", repo_type="model", exist_ok=True, private=True, space_sdk=None
            )
            upload_mock.assert_called_once_with(
                folder_path=cache_path,
                path_in_repo=".",
                repo_id=create_mock.return_value.repo_id,
                repo_type="model",
                revision=None,
                commit_message=None,
                commit_description=None,
                create_pr=False,
                allow_patterns=["*.json"],
                ignore_patterns=None,
                delete_patterns=["*.json"],
            )

    @patch("huggingface_hub.commands.upload.HfApi.repo_info")
    @patch("huggingface_hub.commands.upload.HfApi.upload_file")
    @patch("huggingface_hub.commands.upload.HfApi.create_repo")
    def test_upload_file_mock(self, create_mock: Mock, upload_mock: Mock, repo_info_mock: Mock) -> None:
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
                repo_id="my-dataset", repo_type="dataset", exist_ok=True, private=False, space_sdk=None
            )
            upload_mock.assert_called_once_with(
                path_or_fileobj=str(file_path),
                path_in_repo="logs/file.txt",
                repo_id=create_mock.return_value.repo_id,
                repo_type="dataset",
                revision=None,
                commit_message=None,
                commit_description=None,
                create_pr=True,
            )

    @patch("huggingface_hub.commands.upload.HfApi.repo_info")
    @patch("huggingface_hub.commands.upload.HfApi.upload_file")
    @patch("huggingface_hub.commands.upload.HfApi.create_repo")
    def test_upload_file_no_revision_mock(self, create_mock: Mock, upload_mock: Mock, repo_info_mock: Mock) -> None:
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-model", str(file_path), "logs/file.txt"]))
            cmd.run()
            # Revision not specified => no need to check
            repo_info_mock.assert_not_called()

    @patch("huggingface_hub.commands.upload.HfApi.create_branch")
    @patch("huggingface_hub.commands.upload.HfApi.repo_info")
    @patch("huggingface_hub.commands.upload.HfApi.upload_file")
    @patch("huggingface_hub.commands.upload.HfApi.create_repo")
    def test_upload_file_with_revision_mock(
        self, create_mock: Mock, upload_mock: Mock, repo_info_mock: Mock, create_branch_mock: Mock
    ) -> None:
        repo_info_mock.side_effect = RevisionNotFoundError("revision not found")

        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            cmd = UploadCommand(
                self.parser.parse_args(
                    ["upload", "my-model", str(file_path), "logs/file.txt", "--revision", "my-branch"]
                )
            )
            cmd.run()

            # Revision specified => check that it exists
            repo_info_mock.assert_called_once_with(
                repo_id=create_mock.return_value.repo_id, repo_type="model", revision="my-branch"
            )

            # Revision does not exist => create it
            create_branch_mock.assert_called_once_with(
                repo_id=create_mock.return_value.repo_id, repo_type="model", branch="my-branch", exist_ok=True
            )

    @patch("huggingface_hub.commands.upload.HfApi.repo_info")
    @patch("huggingface_hub.commands.upload.HfApi.upload_file")
    @patch("huggingface_hub.commands.upload.HfApi.create_repo")
    def test_upload_file_revision_and_create_pr_mock(
        self, create_mock: Mock, upload_mock: Mock, repo_info_mock: Mock
    ) -> None:
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            cmd = UploadCommand(
                self.parser.parse_args(
                    ["upload", "my-model", str(file_path), "logs/file.txt", "--revision", "my-branch", "--create-pr"]
                )
            )
            cmd.run()
            # Revision specified but --create-pr => no need to check
            repo_info_mock.assert_not_called()

    @patch("huggingface_hub.commands.upload.HfApi.create_repo")
    def test_upload_missing_path(self, create_mock: Mock) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-model", "/path/to/missing_file", "logs/file.txt"]))
        with self.assertRaises(FileNotFoundError):
            cmd.run()  # File/folder does not exist locally

        # Repo creation happens before the check
        create_mock.assert_not_called()


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
        self.assertIsNone(args.revision)
        self.assertIsNone(args.include)
        self.assertIsNone(args.exclude)
        self.assertIsNone(args.cache_dir)
        self.assertIsNone(args.local_dir)
        self.assertFalse(args.force_download)
        self.assertFalse(args.resume_download)
        self.assertIsNone(args.token)
        self.assertFalse(args.quiet)
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
                "--local-dir",
                ".",
            ]
        )
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.repo_type, "dataset")
        self.assertEqual(args.revision, "v1.0.0")
        self.assertEqual(args.include, ["*.json", "*.yaml"])
        self.assertEqual(args.exclude, ["*.log", "*.txt"])
        self.assertTrue(args.force_download)
        self.assertEqual(args.cache_dir, "/tmp")
        self.assertEqual(args.local_dir, ".")
        self.assertTrue(args.resume_download)
        self.assertEqual(args.token, "my-token")
        self.assertTrue(args.quiet)
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
            local_dir=".",
            local_dir_use_symlinks=None,
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
            resume_download=None,
            force_download=False,
            token="hf_****",
            local_dir=".",
            library_name="huggingface-cli",
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
            local_dir="/path/to/dir",
            local_dir_use_symlinks=None,
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
            local_dir="/path/to/dir",
            library_name="huggingface-cli",
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
            local_dir=None,
            local_dir_use_symlinks=None,
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
            local_dir=None,
            token=None,
            library_name="huggingface-cli",
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
            local_dir=None,
            local_dir_use_symlinks=None,
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
            local_dir=None,
            library_name="huggingface-cli",
        )

        # Same but quiet (no warnings)
        args.quiet = True
        with warnings.catch_warnings():
            # Taken from https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
            warnings.simplefilter("error")
            DownloadCommand(args).run()


class TestTagCommands(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.
        """
        self.parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        TagCommands.register_subcommand(commands_parser)

    def test_tag_create_basic(self) -> None:
        args = self.parser.parse_args(["tag", DUMMY_MODEL_ID, "1.0", "-m", "My tag message"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.tag, "1.0")
        self.assertIsNotNone(args.message)
        self.assertIsNone(args.revision)
        self.assertIsNone(args.token)
        self.assertEqual(args.repo_type, "model")
        self.assertFalse(args.yes)

    def test_tag_create_with_all_options(self) -> None:
        args = self.parser.parse_args(
            [
                "tag",
                DUMMY_MODEL_ID,
                "1.0",
                "--message",
                "My tag message",
                "--revision",
                "v1.0.0",
                "--token",
                "my-token",
                "--repo-type",
                "dataset",
                "--yes",
            ]
        )
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.tag, "1.0")
        self.assertEqual(args.message, "My tag message")
        self.assertEqual(args.revision, "v1.0.0")
        self.assertEqual(args.token, "my-token")
        self.assertEqual(args.repo_type, "dataset")
        self.assertTrue(args.yes)

    def test_tag_list_basic(self) -> None:
        args = self.parser.parse_args(["tag", "--list", DUMMY_MODEL_ID])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertIsNone(args.token)
        self.assertEqual(args.repo_type, "model")

    def test_tag_delete_basic(self) -> None:
        args = self.parser.parse_args(["tag", "--delete", DUMMY_MODEL_ID, "1.0"])
        self.assertEqual(args.repo_id, DUMMY_MODEL_ID)
        self.assertEqual(args.tag, "1.0")
        self.assertIsNone(args.token)
        self.assertEqual(args.repo_type, "model")
        self.assertFalse(args.yes)


@contextmanager
def tmp_current_directory() -> Generator[str, None, None]:
    """Change current directory to a tmp dir and revert back when exiting."""
    with SoftTemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            yield tmp_dir
        except:
            raise
        finally:
            os.chdir(cwd)


class TestRepoFilesCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.
        """
        self.parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        RepoFilesCommand.register_subcommand(commands_parser)

    @patch("huggingface_hub.commands.repo_files.HfApi.delete_files")
    def test_delete(self, delete_files_mock: Mock) -> None:
        fixtures = [
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "*",
                ],
                "delete_files_args": {
                    "delete_patterns": [
                        "*",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file.txt",
                ],
                "delete_files_args": {
                    "delete_patterns": [
                        "file.txt",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "folder/",
                ],
                "delete_files_args": {
                    "delete_patterns": [
                        "folder/",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file1.txt",
                    "folder/",
                    "file2.txt",
                ],
                "delete_files_args": {
                    "delete_patterns": [
                        "file1.txt",
                        "folder/",
                        "file2.txt",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file.txt *",
                    "*.json",
                    "folder/*.parquet",
                ],
                "delete_files_args": {
                    "delete_patterns": [
                        "file.txt *",
                        "*.json",
                        "folder/*.parquet",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            },
            {
                "input_args": [
                    "repo-files",
                    DUMMY_MODEL_ID,
                    "delete",
                    "file.txt *",
                    "--revision",
                    "test_revision",
                    "--repo-type",
                    "dataset",
                    "--commit-message",
                    "My commit message",
                    "--commit-description",
                    "My commit description",
                    "--create-pr",
                ],
                "delete_files_args": {
                    "delete_patterns": [
                        "file.txt *",
                    ],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "dataset",
                    "revision": "test_revision",
                    "commit_message": "My commit message",
                    "commit_description": "My commit description",
                    "create_pr": True,
                },
            },
        ]

        for expected in fixtures:
            # subTest is similar to pytest.mark.parametrize, but using the unittest
            # framework
            with self.subTest(expected):
                delete_files_args = expected["delete_files_args"]

                cmd = DeleteFilesSubCommand(self.parser.parse_args(expected["input_args"]))
                cmd.run()

                if delete_files_args is None:
                    assert delete_files_mock.call_count == 0
                else:
                    assert delete_files_mock.call_count == 1
                    # Inspect the captured calls
                    _, kwargs = delete_files_mock.call_args_list[0]
                    assert kwargs == delete_files_args

                delete_files_mock.reset_mock()
