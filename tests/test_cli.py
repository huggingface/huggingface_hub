import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from huggingface_hub.cli._cli_utils import RepoType
from huggingface_hub.cli.cache import _CANCEL_DELETION_STR
from huggingface_hub.cli.download import download
from huggingface_hub.cli.hf import app
from huggingface_hub.cli.upload import _resolve_upload_paths, upload
from huggingface_hub.errors import RevisionNotFoundError
from huggingface_hub.utils import SoftTemporaryDirectory

from .testing_utils import DUMMY_MODEL_ID


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCacheCommand:
    def test_scan_cache_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.cache.scan_cache_dir") as mock_run:
            result = runner.invoke(app, ["cache", "scan"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(None)

    def test_scan_cache_verbose(self, runner: CliRunner) -> None:
        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir") as mock_run,
            patch("huggingface_hub.cli.cache.get_table") as get_table_mock,
        ):
            result = runner.invoke(app, ["cache", "scan", "-v"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(None)
        get_table_mock.assert_called_once_with(mock_run.return_value, verbosity=1)

    def test_scan_cache_with_dir(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.cache.scan_cache_dir") as mock_run:
            result = runner.invoke(app, ["cache", "scan", "--dir", "something"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with("something")

    def test_scan_cache_ultra_verbose(self, runner: CliRunner) -> None:
        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir") as mock_run,
            patch("huggingface_hub.cli.cache.get_table") as get_table_mock,
        ):
            result = runner.invoke(app, ["cache", "scan", "-vvv"])
        assert result.exit_code == 0
        mock_run.assert_called_once_with(None)
        get_table_mock.assert_called_once_with(mock_run.return_value, verbosity=3)

    def test_delete_cache_with_dir(self, runner: CliRunner) -> None:
        hf_cache_info = Mock()
        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir", return_value=hf_cache_info) as scan_mock,
            patch(
                "huggingface_hub.cli.cache._manual_review_tui",
                return_value=[_CANCEL_DELETION_STR],
            ) as review_mock,
        ):
            result = runner.invoke(app, ["cache", "delete", "--dir", "something"])
        assert result.exit_code == 0
        scan_mock.assert_called_once_with("something")
        review_mock.assert_called_once_with(hf_cache_info, preselected=[], sort_by=None)


class TestUploadCommand:
    def test_upload_basic(self, runner: CliRunner) -> None:
        with SoftTemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "my-folder"
            folder.mkdir()
            with (
                patch(
                    "huggingface_hub.cli.upload._resolve_upload_paths",
                    return_value=(folder.as_posix(), ".", None),
                ) as resolve_mock,
                patch("huggingface_hub.cli.upload.get_hf_api") as api_cls,
            ):
                api = api_cls.return_value
                api.create_repo.return_value = Mock(repo_id=DUMMY_MODEL_ID)
                api.upload_folder.return_value = "uploaded"
                result = runner.invoke(app, ["upload", DUMMY_MODEL_ID, "my-folder"])
        assert result.exit_code == 0
        assert "uploaded" in result.stdout
        resolve_mock.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            local_path="my-folder",
            path_in_repo=None,
            include=None,
        )
        api_cls.assert_called_once_with(token=None)
        api.create_repo.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            repo_type="model",
            exist_ok=True,
            private=False,
            space_sdk=None,
        )
        api.upload_folder.assert_called_once_with(
            folder_path=folder.as_posix(),
            path_in_repo=".",
            repo_id=DUMMY_MODEL_ID,
            repo_type="model",
            revision=None,
            commit_message=None,
            commit_description=None,
            create_pr=False,
            allow_patterns=None,
            ignore_patterns=None,
            delete_patterns=None,
        )

    def test_upload_with_all_options(self, runner: CliRunner) -> None:
        with SoftTemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "my-file"
            folder.mkdir()
            returned_paths = (folder.as_posix(), "data/", ["*.json", "*.yaml"])
            with (
                patch(
                    "huggingface_hub.cli.upload._resolve_upload_paths",
                    return_value=returned_paths,
                ) as resolve_mock,
                patch("huggingface_hub.cli.upload.get_hf_api") as api_cls,
                patch("huggingface_hub.cli.upload.CommitScheduler") as scheduler_cls,
                patch("huggingface_hub.cli.upload.time.sleep", side_effect=KeyboardInterrupt),
            ):
                api = api_cls.return_value
                scheduler = scheduler_cls.return_value
                scheduler.repo_id = DUMMY_MODEL_ID
                result = runner.invoke(
                    app,
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
                        "--include",
                        "*.yaml",
                        "--exclude",
                        "*.log",
                        "--exclude",
                        "*.txt",
                        "--delete",
                        "*.config",
                        "--delete",
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
                    ],
                )
        assert result.exit_code == 0
        assert "Stopped scheduled commits." in result.stdout
        resolve_mock.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            local_path="my-file",
            path_in_repo="data/",
            include=["*.json", "*.yaml"],
        )
        api_cls.assert_called_once_with(token="my-token")
        scheduler_cls.assert_called_once_with(
            folder_path=folder.as_posix(),
            repo_id=DUMMY_MODEL_ID,
            repo_type="dataset",
            revision="v1.0.0",
            allow_patterns=["*.json", "*.yaml"],
            ignore_patterns=["*.log", "*.txt"],
            path_in_repo="data/",
            private=False,
            every=5,
            hf_api=api,
        )
        scheduler.stop.assert_called_once_with()

    def test_every_must_be_positive(self) -> None:
        class _PatchedBadParameter(typer.BadParameter):
            def __init__(self, message: str, *, param_name: Optional[str] = None, **kwargs: object) -> None:
                super().__init__(message, **kwargs)

        with (
            patch("huggingface_hub.cli.upload.typer.BadParameter", _PatchedBadParameter),
            patch("huggingface_hub.cli.upload.get_hf_api") as api_cls,
        ):
            with pytest.raises(typer.BadParameter, match="--every must be a positive value"):
                upload(repo_id=DUMMY_MODEL_ID, every=0)

            with pytest.raises(typer.BadParameter, match="--every must be a positive value"):
                upload(repo_id=DUMMY_MODEL_ID, every=-10)
        api_cls.assert_not_called()

    def test_every_as_int(self, runner: CliRunner) -> None:
        with SoftTemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            with (
                patch(
                    "huggingface_hub.cli.upload._resolve_upload_paths",
                    return_value=(folder.as_posix(), ".", None),
                ),
                patch("huggingface_hub.cli.upload.get_hf_api"),
                patch("huggingface_hub.cli.upload.CommitScheduler") as scheduler_cls,
                patch("huggingface_hub.cli.upload.time.sleep", side_effect=KeyboardInterrupt),
            ):
                result = runner.invoke(app, ["upload", DUMMY_MODEL_ID, ".", "--every", "10"])
        assert result.exit_code == 0
        assert scheduler_cls.call_args.kwargs["every"] == pytest.approx(10)

    def test_every_as_float(self, runner: CliRunner) -> None:
        with SoftTemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            with (
                patch(
                    "huggingface_hub.cli.upload._resolve_upload_paths",
                    return_value=(folder.as_posix(), ".", None),
                ),
                patch("huggingface_hub.cli.upload.get_hf_api"),
                patch("huggingface_hub.cli.upload.CommitScheduler") as scheduler_cls,
                patch("huggingface_hub.cli.upload.time.sleep", side_effect=KeyboardInterrupt),
            ):
                result = runner.invoke(app, ["upload", DUMMY_MODEL_ID, ".", "--every", "0.5"])
        assert result.exit_code == 0
        assert scheduler_cls.call_args.kwargs["every"] == pytest.approx(0.5)


class TestResolveUploadPaths:
    def test_upload_with_wildcard(self) -> None:
        local_path, path_in_repo, include = _resolve_upload_paths(
            repo_id=DUMMY_MODEL_ID, local_path="*.safetensors", path_in_repo=None, include=None
        )
        assert local_path == "."
        assert path_in_repo == "*.safetensors"
        assert include == ["."]

        local_path, path_in_repo, include = _resolve_upload_paths(
            repo_id=DUMMY_MODEL_ID, local_path="subdir/*.safetensors", path_in_repo=None, include=None
        )
        assert local_path == "."
        assert path_in_repo == "subdir/*.safetensors"
        assert include == ["."]

        with pytest.raises(ValueError):
            _resolve_upload_paths(
                repo_id=DUMMY_MODEL_ID,
                local_path="*.safetensors",
                path_in_repo=None,
                include=["*.json"],
            )

        with pytest.raises(ValueError):
            _resolve_upload_paths(
                repo_id=DUMMY_MODEL_ID,
                local_path="*.safetensors",
                path_in_repo="models/",
                include=None,
            )

    def test_upload_implicit_local_path_when_folder_exists(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.mkdir()
            local_path, path_in_repo, include = _resolve_upload_paths(
                repo_id="my-cool-model", local_path=None, path_in_repo=None, include=None
            )
        assert local_path == "my-cool-model"
        assert path_in_repo == "."
        assert include is None

    def test_upload_implicit_local_path_when_file_exists(self) -> None:
        with tmp_current_directory() as cache_dir:
            file_path = Path(cache_dir) / "my-cool-model"
            file_path.write_text("content")
            local_path, path_in_repo, include = _resolve_upload_paths(
                repo_id="my-cool-model", local_path=None, path_in_repo=None, include=None
            )
        assert local_path == "my-cool-model"
        assert path_in_repo == "my-cool-model"
        assert include is None

    def test_upload_implicit_local_path_when_org_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "my-cool-model"
            folder_path.mkdir()
            local_path, path_in_repo, include = _resolve_upload_paths(
                repo_id="my-cool-org/my-cool-model", local_path=None, path_in_repo=None, include=None
            )
        assert local_path == "my-cool-model"
        assert path_in_repo == "."
        assert include is None

    def test_upload_implicit_local_path_otherwise(self) -> None:
        with tmp_current_directory():
            with pytest.raises(ValueError):
                _resolve_upload_paths(repo_id="my-cool-model", local_path=None, path_in_repo=None, include=None)

    def test_upload_explicit_local_path_to_folder_implicit_path_in_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            folder_path = Path(cache_dir) / "path" / "to" / "folder"
            folder_path.mkdir(parents=True, exist_ok=True)
            local_path, path_in_repo, include = _resolve_upload_paths(
                repo_id="my-repo", local_path="./path/to/folder", path_in_repo=None, include=None
            )
        assert local_path == "./path/to/folder"
        assert path_in_repo == "."
        assert include is None

    def test_upload_explicit_local_path_to_file_implicit_path_in_repo(self) -> None:
        with tmp_current_directory() as cache_dir:
            file_path = Path(cache_dir) / "path" / "to" / "file.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("content")
            local_path, path_in_repo, include = _resolve_upload_paths(
                repo_id="my-repo", local_path="./path/to/file.txt", path_in_repo=None, include=None
            )
        assert local_path == "./path/to/file.txt"
        assert path_in_repo == "file.txt"
        assert include is None

    def test_upload_explicit_paths(self) -> None:
        local_path, path_in_repo, include = _resolve_upload_paths(
            repo_id="my-repo", local_path="./path/to/folder", path_in_repo="data/", include=None
        )
        assert local_path == "./path/to/folder"
        assert path_in_repo == "data/"
        assert include is None


class TestUploadImpl:
    @patch("huggingface_hub.cli.upload.is_xet_available", return_value=True)
    @patch("huggingface_hub.cli.upload.HF_HUB_ENABLE_HF_TRANSFER", False)
    def test_upload_folder_mock(self, *_: object) -> None:
        api = Mock()
        api.create_repo.return_value = Mock(repo_id="my-model")
        api.upload_folder.return_value = "done"
        with SoftTemporaryDirectory() as cache_dir:
            cache_path = cache_dir.absolute().as_posix()
            local_dir = Path(cache_path)
            (local_dir / "config.json").write_text("{}")
            with (
                patch("huggingface_hub.cli.upload.get_hf_api", return_value=api),
                patch("builtins.print") as print_mock,
            ):
                upload(
                    repo_id="my-model",
                    local_path=cache_path,
                    include=["*.json"],
                    delete=["*.json"],
                    private=True,
                    quiet=True,
                )
        api.create_repo.assert_called_once_with(
            repo_id="my-model",
            repo_type="model",
            exist_ok=True,
            private=True,
            space_sdk=None,
        )
        api.upload_folder.assert_called_once_with(
            folder_path=cache_path,
            path_in_repo=".",
            repo_id="my-model",
            repo_type="model",
            revision=None,
            commit_message=None,
            commit_description=None,
            create_pr=False,
            allow_patterns=["*.json"],
            ignore_patterns=None,
            delete_patterns=["*.json"],
        )
        print_mock.assert_called_once_with("done")

    @patch("huggingface_hub.cli.upload.is_xet_available", return_value=True)
    @patch("huggingface_hub.cli.upload.HF_HUB_ENABLE_HF_TRANSFER", False)
    def test_upload_file_mock(self, *_: object) -> None:
        api = Mock()
        api.create_repo.return_value = Mock(repo_id="my-dataset")
        api.upload_file.return_value = "uploaded"
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            with (
                patch("huggingface_hub.cli.upload.get_hf_api", return_value=api),
                patch("builtins.print") as print_mock,
            ):
                upload(
                    repo_id="my-dataset",
                    repo_type=RepoType.dataset,
                    local_path=str(file_path),
                    path_in_repo="logs/file.txt",
                    create_pr=True,
                    quiet=True,
                )
        api.create_repo.assert_called_once_with(
            repo_id="my-dataset",
            repo_type="dataset",
            exist_ok=True,
            private=False,
            space_sdk=None,
        )
        api.upload_file.assert_called_once_with(
            path_or_fileobj=str(file_path),
            path_in_repo="logs/file.txt",
            repo_id="my-dataset",
            repo_type="dataset",
            revision=None,
            commit_message=None,
            commit_description=None,
            create_pr=True,
        )
        print_mock.assert_called_once_with("uploaded")

    @patch("huggingface_hub.cli.upload.is_xet_available", return_value=True)
    @patch("huggingface_hub.cli.upload.HF_HUB_ENABLE_HF_TRANSFER", False)
    def test_upload_file_no_revision_mock(self, *_: object) -> None:
        api = Mock()
        api.create_repo.return_value = Mock(repo_id="my-model")
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            with (
                patch("huggingface_hub.cli.upload.get_hf_api", return_value=api),
                patch("builtins.print"),
            ):
                upload(
                    repo_id="my-model",
                    local_path=str(file_path),
                    path_in_repo="logs/file.txt",
                    quiet=True,
                )
        api.repo_info.assert_not_called()

    @patch("huggingface_hub.cli.upload.is_xet_available", return_value=True)
    @patch("huggingface_hub.cli.upload.HF_HUB_ENABLE_HF_TRANSFER", False)
    def test_upload_file_with_revision_mock(self, *_: object) -> None:
        api = Mock()
        api.create_repo.return_value = Mock(repo_id="my-model")
        api.repo_info.side_effect = RevisionNotFoundError("revision not found", response=Mock())
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            with (
                patch("huggingface_hub.cli.upload.get_hf_api", return_value=api),
                patch("builtins.print"),
            ):
                upload(
                    repo_id="my-model",
                    revision="my-branch",
                    local_path=str(file_path),
                    path_in_repo="logs/file.txt",
                    quiet=True,
                )
        api.repo_info.assert_called_once_with(repo_id="my-model", repo_type="model", revision="my-branch")
        api.create_branch.assert_called_once_with(
            repo_id="my-model", repo_type="model", branch="my-branch", exist_ok=True
        )

    @patch("huggingface_hub.cli.upload.is_xet_available", return_value=True)
    @patch("huggingface_hub.cli.upload.HF_HUB_ENABLE_HF_TRANSFER", False)
    def test_upload_file_revision_and_create_pr_mock(self, *_: object) -> None:
        api = Mock()
        api.create_repo.return_value = Mock(repo_id="my-model")
        with SoftTemporaryDirectory() as cache_dir:
            file_path = Path(cache_dir) / "file.txt"
            file_path.write_text("content")
            with (
                patch("huggingface_hub.cli.upload.get_hf_api", return_value=api),
                patch("builtins.print"),
            ):
                upload(
                    repo_id="my-model",
                    revision="my-branch",
                    local_path=str(file_path),
                    path_in_repo="logs/file.txt",
                    create_pr=True,
                    quiet=True,
                )
        api.repo_info.assert_not_called()
        api.create_branch.assert_not_called()

    @patch("huggingface_hub.cli.upload.is_xet_available", return_value=True)
    @patch("huggingface_hub.cli.upload.HF_HUB_ENABLE_HF_TRANSFER", False)
    def test_upload_missing_path(self, *_: object) -> None:
        api = Mock()
        with pytest.raises(FileNotFoundError):
            with patch("huggingface_hub.cli.upload.get_hf_api", return_value=api):
                upload(
                    repo_id="my-model",
                    local_path="/path/to/missing_file",
                    path_in_repo="logs/file.txt",
                    quiet=True,
                )
        api.create_repo.assert_not_called()


class TestDownloadCommand:
    def test_download_basic(self, runner: CliRunner) -> None:
        with (
            patch("huggingface_hub.cli.download.snapshot_download", return_value="path") as snapshot_mock,
            patch("huggingface_hub.cli.download.hf_hub_download") as download_mock,
        ):
            result = runner.invoke(app, ["download", DUMMY_MODEL_ID])
        assert result.exit_code == 0
        assert "path" in result.stdout
        download_mock.assert_not_called()
        snapshot_mock.assert_called_once()
        kwargs = snapshot_mock.call_args.kwargs
        assert kwargs["repo_id"] == DUMMY_MODEL_ID
        assert kwargs["repo_type"] == "model"
        assert kwargs["revision"] is None
        assert kwargs["allow_patterns"] is None
        assert kwargs["ignore_patterns"] is None
        assert kwargs["force_download"] is False
        assert kwargs["cache_dir"] is None
        assert kwargs["local_dir"] is None
        assert kwargs["token"] is None
        assert kwargs["library_name"] == "hf"
        assert kwargs["max_workers"] == 8

    def test_download_with_all_options(self, runner: CliRunner) -> None:
        with (
            patch("huggingface_hub.cli.download.snapshot_download", return_value="path") as snapshot_mock,
            patch("huggingface_hub.cli.download.hf_hub_download") as download_mock,
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    DUMMY_MODEL_ID,
                    "--repo-type",
                    "dataset",
                    "--revision",
                    "v1.0.0",
                    "--include",
                    "*.json",
                    "--include",
                    "*.yaml",
                    "--exclude",
                    "*.log",
                    "--exclude",
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
                ],
            )
        assert result.exit_code == 0
        download_mock.assert_not_called()
        snapshot_mock.assert_called_once()
        kwargs = snapshot_mock.call_args.kwargs
        assert kwargs["repo_id"] == DUMMY_MODEL_ID
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "v1.0.0"
        assert kwargs["allow_patterns"] == ["*.json", "*.yaml"]
        assert kwargs["ignore_patterns"] == ["*.log", "*.txt"]
        assert kwargs["force_download"] is True
        assert kwargs["cache_dir"] == "/tmp"
        assert kwargs["local_dir"] == "."
        assert kwargs["token"] == "my-token"
        assert kwargs["library_name"] == "hf"
        assert kwargs["max_workers"] == 4


class TestDownloadImpl:
    @patch("huggingface_hub.cli.download.snapshot_download")
    @patch("huggingface_hub.cli.download.hf_hub_download")
    def test_download_file_from_revision(self, mock_download: Mock, mock_snapshot: Mock) -> None:
        mock_download.return_value = "file-path"
        with patch("builtins.print") as print_mock:
            download(
                repo_id="author/model",
                filenames=["config.json"],
                repo_type=RepoType.model,
                revision="main",
                quiet=True,
            )
        print_mock.assert_called_once_with("file-path")
        mock_download.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision="main",
            filename="config.json",
            cache_dir=None,
            force_download=False,
            token=None,
            local_dir=None,
            library_name="hf",
        )
        mock_snapshot.assert_not_called()

    @patch("huggingface_hub.cli.download.snapshot_download")
    @patch("huggingface_hub.cli.download.hf_hub_download")
    def test_download_multiple_files(self, mock_download: Mock, mock_snapshot: Mock) -> None:
        mock_snapshot.return_value = "folder-path"
        with patch("builtins.print") as print_mock:
            download(
                repo_id="author/model",
                filenames=["README.md", "config.json"],
                repo_type=RepoType.model,
                force_download=True,
                max_workers=4,
                quiet=True,
            )
        print_mock.assert_called_once_with("folder-path")
        mock_download.assert_not_called()
        mock_snapshot.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["README.md", "config.json"],
            ignore_patterns=None,
            force_download=True,
            cache_dir=None,
            token=None,
            local_dir=None,
            library_name="hf",
            max_workers=4,
        )

    @patch("huggingface_hub.cli.download.snapshot_download")
    def test_download_with_patterns(self, mock_snapshot: Mock) -> None:
        with patch("builtins.print"):
            download(
                repo_id="author/model",
                filenames=[],
                repo_type=RepoType.model,
                include=["*.json"],
                exclude=["data/*"],
                force_download=True,
                quiet=True,
            )
        mock_snapshot.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["*.json"],
            ignore_patterns=["data/*"],
            force_download=True,
            cache_dir=None,
            token=None,
            local_dir=None,
            library_name="hf",
            max_workers=8,
        )

    @patch("huggingface_hub.cli.download.snapshot_download")
    @patch("huggingface_hub.cli.download.hf_hub_download")
    def test_download_with_ignored_patterns(self, mock_download: Mock, mock_snapshot: Mock) -> None:
        mock_snapshot.return_value = "folder-path"
        with (
            patch("builtins.print") as print_mock,
            patch("huggingface_hub.cli.download.logging.set_verbosity_info"),
            patch("huggingface_hub.cli.download.logging.set_verbosity_warning"),
            warnings.catch_warnings(record=True) as caught,
        ):
            download(
                repo_id="author/model",
                filenames=["README.md", "config.json"],
                repo_type=RepoType.model,
                include=["*.json"],
                exclude=["data/*"],
                force_download=True,
            )
        print_mock.assert_called_once_with("folder-path")
        assert any("Ignoring" in str(w.message) for w in caught)
        mock_download.assert_not_called()
        mock_snapshot.assert_called_once_with(
            repo_id="author/model",
            repo_type="model",
            revision=None,
            allow_patterns=["README.md", "config.json"],
            ignore_patterns=None,
            force_download=True,
            cache_dir=None,
            token=None,
            local_dir=None,
            library_name="hf",
            max_workers=8,
        )


class TestTagCommands:
    def test_tag_create_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
                ["repo", "tag", "create", DUMMY_MODEL_ID, "1.0", "-m", "My tag message"],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.create_tag.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            tag="1.0",
            tag_message="My tag message",
            revision=None,
            repo_type="model",
        )

    def test_tag_create_with_all_options(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
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
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token="my-token")
        api.create_tag.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            tag="1.0",
            tag_message="My tag message",
            revision="v1.0.0",
            repo_type="dataset",
        )

    def test_tag_list_basic(self, runner: CliRunner) -> None:
        refs = Mock(tags=[Mock(name="v1")])
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_refs.return_value = refs
            result = runner.invoke(app, ["repo", "tag", "list", DUMMY_MODEL_ID])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.list_repo_refs.assert_called_once_with(repo_id=DUMMY_MODEL_ID, repo_type="model")

    def test_tag_delete_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
                ["repo", "tag", "delete", DUMMY_MODEL_ID, "1.0"],
                input="y\n",
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.delete_tag.assert_called_once_with(repo_id=DUMMY_MODEL_ID, tag="1.0", repo_type="model")


@contextmanager
def tmp_current_directory() -> Generator[str, None, None]:
    with SoftTemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            yield tmp_dir
        finally:
            os.chdir(cwd)


class TestRepoFilesCommand:
    @pytest.mark.parametrize(
        "cli_args, expected_kwargs",
        [
            (
                ["repo-files", "delete", DUMMY_MODEL_ID, "*"],
                {
                    "delete_patterns": ["*"],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            ),
            (
                ["repo-files", "delete", DUMMY_MODEL_ID, "file.txt"],
                {
                    "delete_patterns": ["file.txt"],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            ),
            (
                ["repo-files", "delete", DUMMY_MODEL_ID, "folder/"],
                {
                    "delete_patterns": ["folder/"],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "model",
                    "revision": None,
                    "commit_message": None,
                    "commit_description": None,
                    "create_pr": False,
                },
            ),
            (
                ["repo-files", "delete", DUMMY_MODEL_ID, "file1.txt", "folder/", "file2.txt"],
                {
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
            ),
            (
                [
                    "repo-files",
                    "delete",
                    DUMMY_MODEL_ID,
                    "file.txt *",
                    "*.json",
                    "folder/*.parquet",
                ],
                {
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
            ),
            (
                [
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
                {
                    "delete_patterns": ["file.txt *"],
                    "repo_id": DUMMY_MODEL_ID,
                    "repo_type": "dataset",
                    "revision": "test_revision",
                    "commit_message": "My commit message",
                    "commit_description": "My commit description",
                    "create_pr": True,
                },
            ),
        ],
    )
    def test_delete(self, runner: CliRunner, cli_args: list[str], expected_kwargs: dict[str, object]) -> None:
        with patch("huggingface_hub.cli.repo_files.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(app, cli_args)
        assert result.exit_code == 0
        api.delete_files.assert_called_once_with(**expected_kwargs)


class TestJobsCommand:
    def test_run(self, runner: CliRunner) -> None:
        job = Mock(id="my-job-id", url="https://huggingface.co/api/jobs/my-username/my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
        ):
            api = api_cls.return_value
            api.run_job.return_value = job
            result = runner.invoke(app, ["jobs", "run", "--detach", "ubuntu", "echo", "hello"])
        assert result.exit_code == 0
        api.run_job.assert_called_once_with(
            image="ubuntu",
            command=["echo", "hello"],
            env={},
            secrets={},
            flavor=None,
            timeout=None,
            namespace=None,
        )
        api.fetch_job_logs.assert_not_called()

    def test_create_scheduled_job(self, runner: CliRunner) -> None:
        scheduled_job = Mock(id="my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
        ):
            api = api_cls.return_value
            api.create_scheduled_job.return_value = scheduled_job
            result = runner.invoke(
                app,
                ["jobs", "scheduled", "run", "@hourly", "ubuntu", "echo", "hello"],
            )
        assert result.exit_code == 0
        api.create_scheduled_job.assert_called_once_with(
            image="ubuntu",
            command=["echo", "hello"],
            schedule="@hourly",
            suspend=None,
            concurrency=None,
            env={},
            secrets={},
            flavor=None,
            timeout=None,
            namespace=None,
        )

    def test_uv_command(self, runner: CliRunner) -> None:
        job = Mock(id="my-job-id", url="https://huggingface.co/api/jobs/my-username/my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
        ):
            api = api_cls.return_value
            api.run_uv_job.return_value = job
            result = runner.invoke(app, ["jobs", "uv", "run", "--detach", "echo", "hello"])
        assert result.exit_code == 0
        api.run_uv_job.assert_called_once_with(
            script="echo",
            script_args=["hello"],
            dependencies=None,
            python=None,
            image=None,
            env={},
            secrets={},
            flavor=None,
            timeout=None,
            namespace=None,
            _repo=None,
        )
        api.fetch_job_logs.assert_not_called()

    def test_uv_remote_script(self, runner: CliRunner) -> None:
        job = Mock(id="my-job-id", url="https://huggingface.co/api/jobs/my-username/my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
        ):
            api = api_cls.return_value
            api.run_uv_job.return_value = job
            result = runner.invoke(app, ["jobs", "uv", "run", "--detach", "https://.../script.py"])
        assert result.exit_code == 0
        api.run_uv_job.assert_called_once_with(
            script="https://.../script.py",
            script_args=[],
            dependencies=None,
            python=None,
            image=None,
            env={},
            secrets={},
            flavor=None,
            timeout=None,
            namespace=None,
            _repo=None,
        )

    def test_uv_local_script(self, runner: CliRunner, tmp_path: Path) -> None:
        script_path = tmp_path / "script.py"
        script_path.write_text("print('hello')")
        job = Mock(id="my-job-id", url="https://huggingface.co/api/jobs/my-username/my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
            patch("huggingface_hub.cli.jobs.get_token", return_value="hf_xxx"),
        ):
            api = api_cls.return_value
            api.run_uv_job.return_value = job
            result = runner.invoke(app, ["jobs", "uv", "run", "--detach", str(script_path)])
        assert result.exit_code == 0
        api.run_uv_job.assert_called_once_with(
            script=str(script_path),
            script_args=[],
            dependencies=None,
            python=None,
            image=None,
            env={},
            secrets={},
            flavor=None,
            timeout=None,
            namespace=None,
            _repo=None,
        )
        api.fetch_job_logs.assert_not_called()
