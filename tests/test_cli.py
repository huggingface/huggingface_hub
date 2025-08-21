import os
import unittest
import warnings
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

from huggingface_hub.cli.cache import CacheCommand
from huggingface_hub.cli.download import DownloadCommand
from huggingface_hub.cli.jobs import JobsCommands, RunCommand, ScheduledRunCommand
from huggingface_hub.cli.repo import RepoCommands
from huggingface_hub.cli.repo_files import DeleteFilesSubCommand, RepoFilesCommand
from huggingface_hub.cli.upload import UploadCommand
from huggingface_hub.errors import RevisionNotFoundError
from huggingface_hub.utils import SoftTemporaryDirectory, capture_output

from .testing_utils import DUMMY_MODEL_ID


class TestCacheCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up cache scan/delete commands as in `src/huggingface_hub/cli/hf.py`.
        """
        self.parser = ArgumentParser("hf", usage="hf <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        CacheCommand.register_subcommand(commands_parser)

    def test_scan_cache_basic(self) -> None:
        """Test `hf cache scan`."""
        args = self.parser.parse_args(["cache", "scan"])
        assert args.dir is None
        assert args.verbose == 0
        assert args.func == CacheCommand
        assert args.cache_command == "scan"

    def test_scan_cache_verbose(self) -> None:
        """Test `hf cache scan -v`."""
        args = self.parser.parse_args(["cache", "scan", "-v"])
        assert args.dir is None
        assert args.verbose == 1
        assert args.func == CacheCommand
        assert args.cache_command == "scan"

    def test_scan_cache_with_dir(self) -> None:
        """Test `hf cache scan --dir something`."""
        args = self.parser.parse_args(["cache", "scan", "--dir", "something"])
        assert args.dir == "something"
        assert args.verbose == 0
        assert args.func == CacheCommand
        assert args.cache_command == "scan"

    def test_scan_cache_ultra_verbose(self) -> None:
        """Test `hf cache scan -vvv`."""
        args = self.parser.parse_args(["cache", "scan", "-vvv"])
        assert args.dir is None
        assert args.verbose == 3
        assert args.func == CacheCommand
        assert args.cache_command == "scan"

    def test_delete_cache_with_dir(self) -> None:
        """Test `hf cache delete --dir something`."""
        args = self.parser.parse_args(["cache", "delete", "--dir", "something"])
        assert args.dir == "something"
        assert args.func == CacheCommand
        assert args.cache_command == "delete"


class TestUploadCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/cli/hf.py`.
        """
        self.parser = ArgumentParser("hf", usage="hf <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        UploadCommand.register_subcommand(commands_parser)

    def test_upload_basic(self) -> None:
        """Test `hf upload my-folder to dummy-repo`."""
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "my-folder"]))
        assert cmd.repo_id == DUMMY_MODEL_ID
        assert cmd.local_path == "my-folder"
        assert cmd.path_in_repo == "."  # implicit
        assert cmd.repo_type == "model"
        assert cmd.revision is None
        assert cmd.include is None
        assert cmd.exclude is None
        assert cmd.delete is None
        assert cmd.commit_message is None
        assert cmd.commit_description is None
        assert cmd.create_pr is False
        assert cmd.every is None
        assert cmd.api.token is None
        assert cmd.quiet is False

    def test_upload_with_wildcard(self) -> None:
        """Test uploading files using wildcard patterns."""
        with tmp_current_directory() as cache_dir:
            # Create test files
            (Path(cache_dir) / "model1.safetensors").touch()
            (Path(cache_dir) / "model2.safetensors").touch()
            (Path(cache_dir) / "model.bin").touch()
            (Path(cache_dir) / "config.json").touch()

            # Test basic wildcard pattern
            cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "*.safetensors"]))
            assert cmd.local_path == "."
            assert cmd.include == "*.safetensors"
            assert cmd.path_in_repo == "."
            assert cmd.repo_id == DUMMY_MODEL_ID

            # Test wildcard pattern with specific directory
            subdir = Path(cache_dir) / "subdir"
            subdir.mkdir()
            (subdir / "special.safetensors").touch()

            cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "subdir/*.safetensors"]))
            assert cmd.local_path == "."
            assert cmd.include == "subdir/*.safetensors"
            assert cmd.path_in_repo == "."

            # Test error when using wildcard with --include
            with self.assertRaises(ValueError):
                UploadCommand(
                    self.parser.parse_args(["upload", DUMMY_MODEL_ID, "*.safetensors", "--include", "*.json"])
                )

            # Test error when using wildcard with explicit path_in_repo
            with self.assertRaises(ValueError):
                UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, "*.safetensors", "models/"]))

    def test_upload_with_all_options(self) -> None:
        """Test `hf upload my-file to dummy-repo with all options selected`."""
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
        assert cmd.repo_id == DUMMY_MODEL_ID
        assert cmd.local_path == "my-file"
        assert cmd.path_in_repo == "data/"
        assert cmd.repo_type == "dataset"
        assert cmd.revision == "v1.0.0"
        assert cmd.include == ["*.json", "*.yaml"]
        assert cmd.exclude == ["*.log", "*.txt"]
        assert cmd.delete == ["*.config", "*.secret"]
        assert cmd.commit_message == "My commit message"
        assert cmd.commit_description == "My commit description"
        assert cmd.create_pr is True
        assert cmd.every == 5
        assert cmd.api.token == "my-token"
        assert cmd.quiet is True

    def test_upload_implicit_local_path_when_folder_exists(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.mkdir()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-cool-model"]))

        # A folder with the same name as the repo exists => upload it at the root of the repo
        assert cmd.local_path == "my-cool-model"
        assert cmd.path_in_repo == "."

    def test_upload_implicit_local_path_when_file_exists(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.touch()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-cool-model"]))

        # A file with the same name as the repo exists => upload it at the root of the repo
        assert cmd.local_path == "my-cool-model"
        assert cmd.path_in_repo == "my-cool-model"

    def test_upload_implicit_local_path_when_org_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.mkdir()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-cool-org/my-cool-model"]))

        # A folder with the same name as the repo exists => upload it at the root of the repo
        assert cmd.local_path == "my-cool-model"
        assert cmd.path_in_repo == "."

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
        assert cmd.local_path == "./path/to/folder"
        assert cmd.path_in_repo == "."  # Always upload the folder at the root of the repo

    def test_upload_explicit_local_path_to_file_implicit_path_in_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            file_path = Path(cache_dir) / "path" / "to" / "file.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo", "./path/to/file.txt"]))
        assert cmd.local_path == "./path/to/file.txt"
        assert cmd.path_in_repo == "file.txt"  # If a file, upload it at the root of the repo and keep name

    def test_upload_explicit_paths(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-repo", "./path/to/folder", "data/"]))
        assert cmd.local_path == "./path/to/folder"
        assert cmd.path_in_repo == "data/"

    def test_every_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "0"]))

        with self.assertRaises(ValueError):
            UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "-10"]))

    def test_every_as_int(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "10"]))
        assert cmd.every == 10

    def test_every_as_float(self) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", DUMMY_MODEL_ID, ".", "--every", "0.5"]))
        assert cmd.every == 0.5

    @patch("huggingface_hub.cli.upload.HfApi.repo_info")
    @patch("huggingface_hub.cli.upload.HfApi.upload_folder")
    @patch("huggingface_hub.cli.upload.HfApi.create_repo")
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

    @patch("huggingface_hub.cli.upload.HfApi.repo_info")
    @patch("huggingface_hub.cli.upload.HfApi.upload_file")
    @patch("huggingface_hub.cli.upload.HfApi.create_repo")
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

    @patch("huggingface_hub.cli.upload.HfApi.repo_info")
    @patch("huggingface_hub.cli.upload.HfApi.upload_file")
    @patch("huggingface_hub.cli.upload.HfApi.create_repo")
    def test_upload_file_no_revision_mock(self, create_mock: Mock, upload_mock: Mock, repo_info_mock: Mock) -> None:
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            cmd = UploadCommand(self.parser.parse_args(["upload", "my-model", str(file_path), "logs/file.txt"]))
            cmd.run()
            # Revision not specified => no need to check
            repo_info_mock.assert_not_called()

    @patch("huggingface_hub.cli.upload.HfApi.create_branch")
    @patch("huggingface_hub.cli.upload.HfApi.repo_info")
    @patch("huggingface_hub.cli.upload.HfApi.upload_file")
    @patch("huggingface_hub.cli.upload.HfApi.create_repo")
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

    @patch("huggingface_hub.cli.upload.HfApi.repo_info")
    @patch("huggingface_hub.cli.upload.HfApi.upload_file")
    @patch("huggingface_hub.cli.upload.HfApi.create_repo")
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

    @patch("huggingface_hub.cli.upload.HfApi.create_repo")
    def test_upload_missing_path(self, create_mock: Mock) -> None:
        cmd = UploadCommand(self.parser.parse_args(["upload", "my-model", "/path/to/missing_file", "logs/file.txt"]))
        with self.assertRaises(FileNotFoundError):
            cmd.run()  # File/folder does not exist locally

        # Repo creation happens before the check
        create_mock.assert_not_called()


class TestDownloadCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/cli/hf.py`.
        """
        self.parser = ArgumentParser("hf", usage="hf <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        DownloadCommand.register_subcommand(commands_parser)

    def test_download_basic(self) -> None:
        """Test `hf download dummy-repo`."""
        args = self.parser.parse_args(["download", DUMMY_MODEL_ID])
        assert args.repo_id == DUMMY_MODEL_ID
        assert len(args.filenames) == 0
        assert args.repo_type == "model"
        assert args.revision is None
        assert args.include is None
        assert args.exclude is None
        assert args.cache_dir is None
        assert args.local_dir is None
        assert args.force_download is False
        assert args.token is None
        assert args.quiet is False
        assert args.func == DownloadCommand

    def test_download_with_all_options(self) -> None:
        """Test `hf download dummy-repo` with all options selected."""
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
                "--token",
                "my-token",
                "--quiet",
                "--local-dir",
                ".",
                "--max-workers",
                "4",
            ]
        )
        assert args.repo_id == DUMMY_MODEL_ID
        assert args.repo_type == "dataset"
        assert args.revision == "v1.0.0"
        assert args.include == ["*.json", "*.yaml"]
        assert args.exclude == ["*.log", "*.txt"]
        assert args.force_download is True
        assert args.cache_dir == "/tmp"
        assert args.local_dir == "."
        assert args.token == "my-token"
        assert args.quiet is True
        assert args.max_workers == 4
        assert args.func == DownloadCommand

    @patch("huggingface_hub.cli.download.hf_hub_download")
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
            cache_dir=None,
            local_dir=".",
            quiet=False,
            max_workers=8,
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
            force_download=False,
            token="hf_****",
            local_dir=".",
            library_name="hf",
        )

    @patch("huggingface_hub.cli.download.snapshot_download")
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
            cache_dir=None,
            local_dir="/path/to/dir",
            quiet=False,
            max_workers=8,
        )
        DownloadCommand(args).run()

        # Use `snapshot_download` to ensure all files comes from same revision
        mock.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["README.md", "config.json"],
            ignore_patterns=None,
            force_download=True,
            cache_dir=None,
            token="hf_****",
            local_dir="/path/to/dir",
            library_name="hf",
            max_workers=8,
        )

    @patch("huggingface_hub.cli.download.snapshot_download")
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
            cache_dir=None,
            quiet=False,
            local_dir=None,
            max_workers=8,
        )
        DownloadCommand(args).run()

        # Use `snapshot_download` to ensure all files comes from same revision
        mock.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["*.json"],
            ignore_patterns=["data/*"],
            force_download=True,
            cache_dir=None,
            local_dir=None,
            token=None,
            library_name="hf",
            max_workers=8,
        )

    @patch("huggingface_hub.cli.download.snapshot_download")
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
            max_workers=8,
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
            force_download=True,
            cache_dir=None,
            token=None,
            local_dir=None,
            library_name="hf",
            max_workers=8,
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
        Set up CLI as in `src/huggingface_hub/cli/hf.py`.
        """
        self.parser = ArgumentParser("hf", usage="hf <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        RepoCommands.register_subcommand(commands_parser)

    def test_tag_create_basic(self) -> None:
        args = self.parser.parse_args(["repo", "tag", "create", DUMMY_MODEL_ID, "1.0", "-m", "My tag message"])
        assert args.repo_id == DUMMY_MODEL_ID
        assert args.tag == "1.0"
        assert args.message is not None
        assert args.revision is None
        assert args.token is None
        assert args.repo_type == "model"

    def test_tag_create_with_all_options(self) -> None:
        args = self.parser.parse_args(
            [
                "repo",
                "tag",
                "create",
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
            ]
        )
        assert args.repo_id == DUMMY_MODEL_ID
        assert args.tag == "1.0"
        assert args.message == "My tag message"
        assert args.revision == "v1.0.0"
        assert args.token == "my-token"
        assert args.repo_type == "dataset"

    def test_tag_list_basic(self) -> None:
        args = self.parser.parse_args(["repo", "tag", "list", DUMMY_MODEL_ID])
        assert args.repo_id == DUMMY_MODEL_ID
        assert args.token is None
        assert args.repo_type == "model"

    def test_tag_delete_basic(self) -> None:
        args = self.parser.parse_args(["repo", "tag", "delete", DUMMY_MODEL_ID, "1.0"])
        assert args.repo_id == DUMMY_MODEL_ID
        assert args.tag == "1.0"
        assert args.token is None
        assert args.repo_type == "model"
        assert args.yes is False


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
        Set up CLI as in `src/huggingface_hub/cli/hf.py`.
        """
        self.parser = ArgumentParser("hf", usage="hf <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        RepoFilesCommand.register_subcommand(commands_parser)

    @patch("huggingface_hub.cli.repo_files.HfApi.delete_files")
    def test_delete(self, delete_files_mock: Mock) -> None:
        fixtures = [
            {
                "input_args": [
                    "repo-files",
                    "delete",
                    DUMMY_MODEL_ID,
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
                    "delete",
                    DUMMY_MODEL_ID,
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
                    "delete",
                    DUMMY_MODEL_ID,
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
                    "delete",
                    DUMMY_MODEL_ID,
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
                    "delete",
                    DUMMY_MODEL_ID,
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
                    "delete",
                    DUMMY_MODEL_ID,
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


class DummyResponse:
    def __init__(self, json):
        self._json = json

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class TestJobsCommand(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up CLI as in `src/huggingface_hub/commands/huggingface_cli.py`.
        """
        self.parser = ArgumentParser("hf", usage="hf <command> [<args>]")
        commands_parser = self.parser.add_subparsers()
        JobsCommands.register_subcommand(commands_parser)

    @patch(
        "requests.Session.post",
        return_value=DummyResponse(
            {
                "id": "my-job-id",
                "owner": {
                    "id": "userid",
                    "name": "my-username",
                    "type": "user",
                },
                "status": {"stage": "RUNNING"},
            }
        ),
    )
    @patch("huggingface_hub.hf_api.HfApi.whoami", return_value={"name": "my-username"})
    def test_run(self, whoami: Mock, requests_post: Mock) -> None:
        input_args = ["jobs", "run", "--detach", "ubuntu", "echo", "hello"]
        cmd = RunCommand(self.parser.parse_args(input_args))
        cmd.run()
        assert requests_post.call_count == 1
        args, kwargs = requests_post.call_args_list[0]
        assert args == ("https://huggingface.co/api/jobs/my-username",)
        assert kwargs["json"] == {
            "command": ["echo", "hello"],
            "arguments": [],
            "environment": {},
            "flavor": "cpu-basic",
            "dockerImage": "ubuntu",
        }

    @patch(
        "requests.Session.post",
        return_value=DummyResponse(
            {
                "id": "my-job-id",
                "owner": {
                    "id": "userid",
                    "name": "my-username",
                    "type": "user",
                },
                "status": {"lastJob": None, "nextJobRunAt": "2025-08-20T15:35:00.000Z"},
                "jobSpec": {},
            }
        ),
    )
    @patch("huggingface_hub.hf_api.HfApi.whoami", return_value={"name": "my-username"})
    def test_schedule(self, whoami: Mock, requests_post: Mock) -> None:
        input_args = ["jobs", "scheduled", "run", "@hourly", "ubuntu", "echo", "hello"]
        cmd = ScheduledRunCommand(self.parser.parse_args(input_args))
        cmd.run()
        assert requests_post.call_count == 1
        args, kwargs = requests_post.call_args_list[0]
        assert args == ("https://huggingface.co/api/scheduled-jobs/my-username",)
        assert kwargs["json"] == {
            "jobSpec": {
                "command": ["echo", "hello"],
                "arguments": [],
                "environment": {},
                "flavor": "cpu-basic",
                "dockerImage": "ubuntu",
            },
            "schedule": "@hourly",
            "suspend": False,
            "concurrency": False,
        }
