import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Generator, Optional
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from huggingface_hub.cli._cli_utils import RepoType
from huggingface_hub.cli.cache import CacheDeletionCounts
from huggingface_hub.cli.download import download
from huggingface_hub.cli.hf import app
from huggingface_hub.cli.upload import _resolve_upload_paths, upload
from huggingface_hub.errors import RevisionNotFoundError
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import (
    CachedFileInfo,
    CachedRepoInfo,
    CachedRevisionInfo,
    HFCacheInfo,
    SoftTemporaryDirectory,
)
from huggingface_hub.utils._verification import FolderVerification

from .testing_utils import DUMMY_MODEL_ID


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_revision(commit_hash: str, *, refs: Optional[set[str]] = None) -> CachedRevisionInfo:
    return CachedRevisionInfo(
        commit_hash=commit_hash,
        snapshot_path=Path(f"/tmp/{commit_hash}"),
        size_on_disk=0,
        files=frozenset(),
        refs=frozenset(refs or set()),
        last_modified=0.0,
    )


def _make_repo(repo_id: str, *, revisions: list[CachedRevisionInfo]) -> CachedRepoInfo:
    return CachedRepoInfo(
        repo_id=repo_id,
        repo_type="model",
        repo_path=Path(f"/tmp/{repo_id.replace('/', '_')}"),
        size_on_disk=0,
        nb_files=0,
        revisions=frozenset(revisions),
        last_accessed=0.0,
        last_modified=0.0,
    )


class TestCacheCommand:
    def test_ls_table_output(self, runner: CliRunner) -> None:
        repo = _make_repo("user/model", revisions=[_make_revision("a" * 40, refs={"main"})])
        entries = [(repo, None)]
        repo_refs_map = {repo: frozenset({"main"})}

        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir"),
            patch(
                "huggingface_hub.cli.cache.collect_cache_entries",
                return_value=(entries, repo_refs_map),
            ),
        ):
            result = runner.invoke(app, ["cache", "ls"])

        assert result.exit_code == 0
        stdout = result.stdout
        assert "model/user/model" in stdout
        assert "main" in stdout

    def test_ls_json_with_filter_and_revisions(self, runner: CliRunner) -> None:
        revision = _make_revision("b" * 40, refs={"main"})
        repo = _make_repo("user/model", revisions=[revision])
        entries = [(repo, revision)]
        repo_refs_map = {repo: frozenset({"main"})}

        def true_filter(repo: CachedRepoInfo, revision_obj: Optional[CachedRevisionInfo], now: float) -> bool:
            return True

        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir"),
            patch(
                "huggingface_hub.cli.cache.collect_cache_entries",
                return_value=(entries, repo_refs_map),
            ),
            patch(
                "huggingface_hub.cli.cache.compile_cache_filter",
                return_value=true_filter,
            ) as compile_mock,
        ):
            result = runner.invoke(
                app,
                ["cache", "ls", "--revisions", "--filter", "size>1", "--format", "json"],
            )

        assert result.exit_code == 0
        compile_mock.assert_called_once_with("size>1", repo_refs_map)
        payload = json.loads(result.stdout)
        assert payload and payload[0]["revision"] == revision.commit_hash

    def test_ls_quiet_revisions(self, runner: CliRunner) -> None:
        revision = _make_revision("c" * 40, refs=set())
        repo = _make_repo("user/model", revisions=[revision])
        entries = [(repo, revision)]
        repo_refs_map = {repo: frozenset()}

        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir"),
            patch(
                "huggingface_hub.cli.cache.collect_cache_entries",
                return_value=(entries, repo_refs_map),
            ),
        ):
            result = runner.invoke(app, ["cache", "ls", "--revisions", "--quiet"])

        assert result.exit_code == 0
        assert result.stdout.strip() == revision.commit_hash

    def test_ls_with_sort(self, runner: CliRunner) -> None:
        repo1 = _make_repo("user/model1", revisions=[_make_revision("d" * 40)])
        repo2 = _make_repo("user/model2", revisions=[_make_revision("e" * 40)])
        repo3 = _make_repo("user/model3", revisions=[_make_revision("f" * 40)])
        entries = [(repo1, None), (repo2, None), (repo3, None)]
        repo_refs_map = {repo1: frozenset(), repo2: frozenset(), repo3: frozenset()}

        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir"),
            patch(
                "huggingface_hub.cli.cache.collect_cache_entries",
                return_value=(entries, repo_refs_map),
            ),
        ):
            result = runner.invoke(app, ["cache", "ls", "--sort", "name:desc", "--limit", "2"])

        assert result.exit_code == 0
        stdout = result.stdout

        # Check alphabetical order
        assert stdout.index("model3") < stdout.index("model2")  # descending order

        # Check limit of 2 entries
        assert "model1" not in stdout

    def test_rm_revision_executes_strategy(self, runner: CliRunner) -> None:
        revision = _make_revision("c" * 40)
        repo = _make_repo("user/model", revisions=[revision])

        repo_lookup = {"model/user/model": repo}
        revision_lookup = {revision.commit_hash.lower(): (repo, revision)}

        strategy = Mock()
        strategy.expected_freed_size_str = "0B"

        hf_cache_info = Mock()
        hf_cache_info.delete_revisions.return_value = strategy

        counts = CacheDeletionCounts(repo_count=0, partial_revision_count=1, total_revision_count=1)

        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir", return_value=hf_cache_info),
            patch("huggingface_hub.cli.cache.build_cache_index", return_value=(repo_lookup, revision_lookup)),
            patch(
                "huggingface_hub.cli.cache.summarize_deletions",
                return_value=counts,
            ),
            patch("huggingface_hub.cli.cache.print_cache_selected_revisions") as print_mock,
        ):
            result = runner.invoke(app, ["cache", "rm", revision.commit_hash, "--yes"])

        assert result.exit_code == 0
        hf_cache_info.delete_revisions.assert_called_once_with(revision.commit_hash)
        strategy.execute.assert_called_once_with()
        print_mock.assert_called_once()

    def test_rm_dry_run_skips_execute(self, runner: CliRunner) -> None:
        revision = _make_revision("d" * 40)
        repo = _make_repo("user/model", revisions=[revision])
        repo_lookup = {"model/user/model": repo}
        revision_lookup = {revision.commit_hash.lower(): (repo, revision)}

        strategy = Mock()
        strategy.expected_freed_size_str = "0B"

        hf_cache_info = Mock()
        hf_cache_info.delete_revisions.return_value = strategy

        counts = CacheDeletionCounts(repo_count=0, partial_revision_count=1, total_revision_count=1)

        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir", return_value=hf_cache_info),
            patch("huggingface_hub.cli.cache.build_cache_index", return_value=(repo_lookup, revision_lookup)),
            patch(
                "huggingface_hub.cli.cache.summarize_deletions",
                return_value=counts,
            ),
            patch("huggingface_hub.cli.cache.print_cache_selected_revisions"),
        ):
            result = runner.invoke(app, ["cache", "rm", revision.commit_hash, "--dry-run"])

        assert result.exit_code == 0
        hf_cache_info.delete_revisions.assert_called_once_with(revision.commit_hash)
        strategy.execute.assert_not_called()

    def test_prune_dry_run(self, runner: CliRunner) -> None:
        referenced = _make_revision("e" * 40, refs={"main"})
        detached = _make_revision("f" * 40, refs=set())
        repo = _make_repo("user/model", revisions=[referenced, detached])

        hf_cache_info = Mock()
        hf_cache_info.repos = frozenset({repo})

        strategy = Mock()
        strategy.expected_freed_size_str = "0B"
        hf_cache_info.delete_revisions.return_value = strategy

        counts = CacheDeletionCounts(repo_count=0, partial_revision_count=1, total_revision_count=1)

        with (
            patch("huggingface_hub.cli.cache.scan_cache_dir", return_value=hf_cache_info),
            patch(
                "huggingface_hub.cli.cache.summarize_deletions",
                return_value=counts,
            ),
            patch("huggingface_hub.cli.cache.print_cache_selected_revisions") as print_mock,
        ):
            result = runner.invoke(app, ["cache", "prune", "--dry-run"])

        assert result.exit_code == 0
        hf_cache_info.delete_revisions.assert_called_once_with(detached.commit_hash)
        strategy.execute.assert_not_called()
        print_mock.assert_called_once()

    def test_verify_success(self, runner: CliRunner) -> None:
        repo_id = "user/model"
        verified_path = Path("/tmp/cache/user/model")
        result_obj = FolderVerification(
            revision="main",
            checked_count=1,
            mismatches=[],
            missing_paths=[],
            extra_paths=[],
            verified_path=verified_path,
        )

        with patch("huggingface_hub.cli.cache.get_hf_api") as get_api_mock:
            api = get_api_mock.return_value
            api.verify_repo_checksums.return_value = result_obj
            result = runner.invoke(app, ["cache", "verify", repo_id])

        assert result.exit_code == 0
        stdout = result.stdout
        normalized_stdout = stdout.replace("\\", "/")
        expected_path_str = verified_path.as_posix()
        assert f"✅ Verified 1 file(s) for 'user/model' (model) in {expected_path_str}" in normalized_stdout
        assert "  All checksums match." in stdout
        get_api_mock.assert_called_once()
        api.verify_repo_checksums.assert_called_once_with(
            repo_id=repo_id,
            repo_type="model",
            revision=None,
            cache_dir=None,
            local_dir=None,
            token=None,
        )

    def test_verify_reports_mismatch(self, runner: CliRunner) -> None:
        repo_id = "user/model"
        result_obj = FolderVerification(
            revision="main",
            checked_count=1,
            mismatches=[{"path": "pytorch_model.bin", "expected": "dead", "actual": "beef", "algorithm": "sha256"}],
            missing_paths=[],
            extra_paths=[],
            verified_path=Path("/tmp/cache/user/model"),
        )

        with patch("huggingface_hub.cli.cache.get_hf_api") as get_api_mock:
            api = get_api_mock.return_value
            api.verify_repo_checksums.return_value = result_obj
            result = runner.invoke(app, ["cache", "verify", repo_id])

        assert result.exit_code == 1
        assert "Checksum verification failed" in result.stdout
        assert "pytorch_model.bin" in result.stdout
        assert "expected" in result.stdout
        assert "Verification failed for 'user/model' (model)" in result.stdout
        assert "Revision: main" in result.stdout

    def test_verify_reports_missing_local_file(self, runner: CliRunner) -> None:
        commit_hash = "4" * 40
        repo_id = "user/model"
        file_name = "config.json"

        with SoftTemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            snapshot_path = base / "snapshots" / commit_hash
            snapshot_path.mkdir(parents=True)

            blob_dir = base / "blobs"
            blob_dir.mkdir()

            blob_path = blob_dir / ("a" * 64)
            blob_path.write_bytes(b"hello")

            file_path = snapshot_path / file_name
            file_path.touch()

            file_info = CachedFileInfo(
                file_name=file_name,
                file_path=file_path,
                blob_path=blob_path,
                size_on_disk=blob_path.stat().st_size,
                blob_last_accessed=0.0,
                blob_last_modified=0.0,
            )
            revision = CachedRevisionInfo(
                commit_hash=commit_hash,
                snapshot_path=snapshot_path,
                size_on_disk=blob_path.stat().st_size,
                files=frozenset({file_info}),
                refs=frozenset({"main"}),
                last_modified=0.0,
            )
            repo = CachedRepoInfo(
                repo_id=repo_id,
                repo_type="model",
                repo_path=base,
                size_on_disk=blob_path.stat().st_size,
                nb_files=1,
                revisions=frozenset({revision}),
                last_accessed=0.0,
                last_modified=0.0,
            )
            hf_cache_info = HFCacheInfo(
                size_on_disk=blob_path.stat().st_size,
                repos=frozenset({repo}),
                warnings=[],
            )

            with (
                patch("huggingface_hub.cli.cache.scan_cache_dir", return_value=hf_cache_info),
                patch("huggingface_hub.cli.cache.get_hf_api") as get_api_mock,
            ):
                api = get_api_mock.return_value
                api.list_repo_tree.return_value = [
                    SimpleNamespace(path=file_name, blob_id="unused", lfs=None),
                    SimpleNamespace(
                        path="missing.txt",
                        blob_id="blobid",
                        lfs=None,
                    ),
                ]
                result = runner.invoke(app, ["cache", "verify", repo.cache_id])

        assert result.exit_code == 1
        assert "missing locally" in result.stdout
        assert "Verification failed for" in result.stdout
        assert "Revision:" in result.stdout


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
            private=None,
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
            private=None,
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
            private=None,
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
        assert kwargs["library_name"] == "huggingface-cli"
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
        assert kwargs["library_name"] == "huggingface-cli"
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
            library_name="huggingface-cli",
            dry_run=False,
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
            library_name="huggingface-cli",
            max_workers=4,
            dry_run=False,
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
            library_name="huggingface-cli",
            max_workers=8,
            dry_run=False,
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
            library_name="huggingface-cli",
            max_workers=8,
            dry_run=False,
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


class TestBranchCommands:
    def test_branch_create_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(app, ["repo", "branch", "create", DUMMY_MODEL_ID, "dev"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.create_branch.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            branch="dev",
            revision=None,
            repo_type="model",
            exist_ok=False,
        )

    def test_branch_create_with_all_options(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
                [
                    "repo",
                    "branch",
                    "create",
                    DUMMY_MODEL_ID,
                    "dev",
                    "--repo-type",
                    "dataset",
                    "--revision",
                    "v1.0.0",
                    "--token",
                    "my-token",
                    "--exist-ok",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token="my-token")
        api.create_branch.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            branch="dev",
            revision="v1.0.0",
            repo_type="dataset",
            exist_ok=True,
        )

    def test_branch_delete_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(app, ["repo", "branch", "delete", DUMMY_MODEL_ID, "dev"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.delete_branch.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            branch="dev",
            repo_type="model",
        )

    def test_branch_delete_with_all_options(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
                [
                    "repo",
                    "branch",
                    "delete",
                    DUMMY_MODEL_ID,
                    "dev",
                    "--repo-type",
                    "dataset",
                    "--token",
                    "my-token",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token="my-token")
        api.delete_branch.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            branch="dev",
            repo_type="dataset",
        )


class TestRepoMoveCommand:
    def test_repo_move_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(app, ["repo", "move", DUMMY_MODEL_ID, "new-id"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.move_repo.assert_called_once_with(
            from_id=DUMMY_MODEL_ID,
            to_id="new-id",
            repo_type="model",
        )

    def test_repo_move_with_all_options(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
                [
                    "repo",
                    "move",
                    DUMMY_MODEL_ID,
                    "new-id",
                    "--repo-type",
                    "dataset",
                    "--token",
                    "my-token",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token="my-token")
        api.move_repo.assert_called_once_with(
            from_id=DUMMY_MODEL_ID,
            to_id="new-id",
            repo_type="dataset",
        )


class TestRepoSettingsCommand:
    def test_repo_settings_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(app, ["repo", "settings", DUMMY_MODEL_ID])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.update_repo_settings.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            gated=None,
            private=None,
            repo_type="model",
        )

    def test_repo_settings_with_all_options(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
                [
                    "repo",
                    "settings",
                    DUMMY_MODEL_ID,
                    "--gated",
                    "manual",
                    "--private",
                    "--repo-type",
                    "dataset",
                    "--token",
                    "my-token",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token="my-token")
        kwargs = api.update_repo_settings.call_args.kwargs
        assert kwargs["repo_id"] == DUMMY_MODEL_ID
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["private"] is True
        assert kwargs["gated"] == "manual"


class TestRepoDeleteCommand:
    def test_repo_delete_basic(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(app, ["repo", "delete", DUMMY_MODEL_ID])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.delete_repo.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            repo_type="model",
            missing_ok=False,
        )

    def test_repo_delete_with_all_options(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repo.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(
                app,
                [
                    "repo",
                    "delete",
                    DUMMY_MODEL_ID,
                    "--repo-type",
                    "dataset",
                    "--token",
                    "my-token",
                    "--missing-ok",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token="my-token")
        api.delete_repo.assert_called_once_with(
            repo_id=DUMMY_MODEL_ID,
            repo_type="dataset",
            missing_ok=True,
        )


class TestModelsLsCommand:
    def test_models_ls_basic(self, runner: CliRunner) -> None:
        repo = ModelInfo(
            id="user/model-id",
            downloads=100,
            likes=50,
            trending_score=10,
            created_at="2025-01-01T12:00:00Z",
            private=False,
            pipeline_tag="text-classification",
            library_name="transformers",
            tags=[],
            siblings=[],
            spaces=[],
            card_data=None,
            last_modified=None,
            config=None,
            transformers_info=None,
        )

        with patch("huggingface_hub.cli.models.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = iter([repo])
            result = runner.invoke(app, ["models", "ls", "--format", "json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output[0]["id"] == "user/model-id"
        assert output[0]["created_at"] == "2025-01-01T12:00:00+00:00"

    def test_models_ls_none_fields_excluded(self, runner: CliRunner) -> None:
        repo = ModelInfo(
            id="user/model-id",
            downloads=None,
            likes=None,
            private=False,
            tags=[],
            siblings=[],
            spaces=[],
        )

        with patch("huggingface_hub.cli.models.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = iter([repo])
            result = runner.invoke(app, ["models", "ls", "--format", "json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert "downloads" not in output[0]
        assert "likes" not in output[0]

    def test_models_ls_with_sort(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.models.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = iter([])
            result = runner.invoke(app, ["models", "ls", "--sort", "likes"])

        assert result.exit_code == 0
        _, kwargs = api.list_models.call_args
        assert kwargs["sort"] == "likes"

    def test_models_ls_with_filters(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.models.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = iter([])
            result = runner.invoke(
                app,
                [
                    "models",
                    "ls",
                    "--author",
                    "google",
                    "--search",
                    "bert",
                    "--filter",
                    "text-classification",
                    "--limit",
                    "5",
                ],
            )

        assert result.exit_code == 0
        _, kwargs = api.list_models.call_args
        assert kwargs["author"] == "google"
        assert kwargs["search"] == "bert"
        assert kwargs["filter"] == ["text-classification"]
        assert kwargs["limit"] == 5

    def test_models_ls_invalid_sort_key(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["models", "ls", "--sort", "invalid_key"])
        assert result.exit_code == 2
        assert "Invalid value" in result.output


class TestDatasetsLsCommand:
    def test_datasets_ls_with_sort(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.datasets.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_datasets.return_value = iter([])
            result = runner.invoke(app, ["datasets", "ls", "--sort", "downloads"])

        assert result.exit_code == 0
        _, kwargs = api.list_datasets.call_args
        assert kwargs["sort"] == "downloads"


class TestSpacesLsCommand:
    def test_spaces_ls(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.spaces.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_spaces.return_value = iter([])
            result = runner.invoke(app, ["spaces", "ls"])

        assert result.exit_code == 0
        api.list_spaces.assert_called_once()

    def test_spaces_ls_downloads_sort_invalid(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["spaces", "ls", "--sort", "downloads"])
        assert result.exit_code == 2
        assert "Invalid value" in result.output


class TestInferenceEndpointsCommands:
    def test_list(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "demo"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_inference_endpoints.return_value = [endpoint]
            result = runner.invoke(app, ["endpoints", "ls", "--format", "json"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.list_inference_endpoints.assert_called_once_with(namespace=None, token=None)
        output = json.loads(result.stdout)
        assert output[0]["name"] == "demo"

    def test_list_with_format_and_quiet(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "demo", "status": {"state": "running"}, "model": {"repository": "user/model"}})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_inference_endpoints.return_value = [endpoint]
            # Test table format
            result = runner.invoke(app, ["endpoints", "ls", "--format", "table"])
        assert result.exit_code == 0
        assert "NAME" in result.stdout
        assert "demo" in result.stdout

        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_inference_endpoints.return_value = [endpoint]
            # Test quiet mode
            result = runner.invoke(app, ["endpoints", "ls", "--quiet"])
        assert result.exit_code == 0
        assert result.stdout.strip() == "demo"

    def test_inference_endpoints_alias(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "alias"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_inference_endpoints.return_value = [endpoint]
            result = runner.invoke(app, ["endpoints", "ls", "--format", "json"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.list_inference_endpoints.assert_called_once_with(namespace=None, token=None)
        output = json.loads(result.stdout)
        assert output[0]["name"] == "alias"

    def test_deploy_from_hub(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "hub"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.create_inference_endpoint.return_value = endpoint
            result = runner.invoke(
                app,
                [
                    "endpoints",
                    "deploy",
                    "my-endpoint",
                    "--repo",
                    "my-repo",
                    "--framework",
                    "custom",
                    "--accelerator",
                    "cpu",
                    "--instance-size",
                    "x4",
                    "--instance-type",
                    "standard",
                    "--region",
                    "us-east-1",
                    "--vendor",
                    "aws",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.create_inference_endpoint.assert_called_once_with(
            name="my-endpoint",
            repository="my-repo",
            framework="custom",
            accelerator="cpu",
            instance_size="x4",
            instance_type="standard",
            region="us-east-1",
            vendor="aws",
            namespace=None,
            token=None,
            task=None,
            min_replica=1,
            max_replica=1,
            scaling_metric=None,
            scaling_threshold=None,
            scale_to_zero_timeout=None,
        )
        assert '"name": "hub"' in result.stdout

    def test_deploy_from_catalog(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "catalog"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.create_inference_endpoint_from_catalog.return_value = endpoint
            result = runner.invoke(
                app,
                [
                    "endpoints",
                    "catalog",
                    "deploy",
                    "--repo",
                    "catalog/model",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.create_inference_endpoint_from_catalog.assert_called_once_with(
            repo_id="catalog/model",
            name=None,
            namespace=None,
            token=None,
        )
        assert '"name": "catalog"' in result.stdout

    def test_describe(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "describe"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.get_inference_endpoint.return_value = endpoint
            result = runner.invoke(app, ["endpoints", "describe", "my-endpoint"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.get_inference_endpoint.assert_called_once_with(name="my-endpoint", namespace=None, token=None)
        assert '"name": "describe"' in result.stdout

    def test_update(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "updated"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.update_inference_endpoint.return_value = endpoint
            result = runner.invoke(
                app,
                [
                    "endpoints",
                    "update",
                    "my-endpoint",
                    "--repo",
                    "my-repo",
                    "--accelerator",
                    "gpu",
                    "--instance-size",
                    "x4",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.update_inference_endpoint.assert_called_once_with(
            name="my-endpoint",
            namespace=None,
            repository="my-repo",
            framework=None,
            revision=None,
            task=None,
            accelerator="gpu",
            instance_size="x4",
            instance_type=None,
            min_replica=None,
            max_replica=None,
            scale_to_zero_timeout=None,
            token=None,
            scaling_metric=None,
            scaling_threshold=None,
        )
        assert '"name": "updated"' in result.stdout

    def test_delete(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            result = runner.invoke(app, ["endpoints", "delete", "my-endpoint", "--yes"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.delete_inference_endpoint.assert_called_once_with(name="my-endpoint", namespace=None, token=None)
        assert "Deleted 'my-endpoint'." in result.stdout

    def test_pause(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "paused"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.pause_inference_endpoint.return_value = endpoint
            result = runner.invoke(app, ["endpoints", "pause", "my-endpoint"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.pause_inference_endpoint.assert_called_once_with(name="my-endpoint", namespace=None, token=None)
        assert '"name": "paused"' in result.stdout

    def test_resume(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "resumed"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.resume_inference_endpoint.return_value = endpoint
            result = runner.invoke(app, ["endpoints", "resume", "my-endpoint"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.resume_inference_endpoint.assert_called_once_with(
            name="my-endpoint",
            namespace=None,
            token=None,
            running_ok=True,
        )
        assert '"name": "resumed"' in result.stdout

    def test_resume_fail_if_already_running(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "resumed"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.resume_inference_endpoint.return_value = endpoint
            result = runner.invoke(
                app,
                [
                    "endpoints",
                    "resume",
                    "my-endpoint",
                    "--fail-if-already-running",
                ],
            )
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.resume_inference_endpoint.assert_called_once_with(
            name="my-endpoint",
            namespace=None,
            token=None,
            running_ok=False,
        )
        assert '"name": "resumed"' in result.stdout

    def test_scale_to_zero(self, runner: CliRunner) -> None:
        endpoint = Mock(raw={"name": "zero"})
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.scale_to_zero_inference_endpoint.return_value = endpoint
            result = runner.invoke(app, ["endpoints", "scale-to-zero", "my-endpoint"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.scale_to_zero_inference_endpoint.assert_called_once_with(
            name="my-endpoint",
            namespace=None,
            token=None,
        )
        assert '"name": "zero"' in result.stdout

    def test_list_catalog(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.inference_endpoints.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_inference_catalog.return_value = ["model"]
            result = runner.invoke(app, ["endpoints", "catalog", "ls"])
        assert result.exit_code == 0
        api_cls.assert_called_once_with(token=None)
        api.list_inference_catalog.assert_called_once_with(token=None)
        assert '"models"' in result.stdout
        assert '"model"' in result.stdout


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
            labels=None,
            flavor=None,
            timeout=None,
            namespace=None,
        )
        api.fetch_job_logs.assert_not_called()

    def test_run_with_extra_args(self, runner: CliRunner) -> None:
        job = Mock(id="my-job-id", url="https://huggingface.co/api/jobs/my-username/my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
        ):
            api = api_cls.return_value
            api.run_job.return_value = job
            result = runner.invoke(
                app, ["jobs", "run", "--detach", "python:3.12", "python", "-c", "'print(\"Hello from the cloud!\")'"]
            )
        assert result.exit_code == 0
        api.run_job.assert_called_once_with(
            image="python:3.12",
            command=["python", "-c", "'print(\"Hello from the cloud!\")'"],
            env={},
            secrets={},
            labels=None,
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
            labels=None,
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
            labels=None,
            flavor=None,
            timeout=None,
            namespace=None,
        )
        api.fetch_job_logs.assert_not_called()

    def test_uv_command_with_extra_args(self, runner: CliRunner) -> None:
        job = Mock(id="my-job-id", url="https://huggingface.co/api/jobs/my-username/my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
        ):
            api = api_cls.return_value
            api.run_uv_job.return_value = job
            result = runner.invoke(
                app, ["jobs", "uv", "run", "--detach", "python", "-c", "'print(\"Hello from the cloud!\")'"]
            )
        assert result.exit_code == 0
        api.run_uv_job.assert_called_once_with(
            script="python",
            script_args=["-c", "'print(\"Hello from the cloud!\")'"],
            dependencies=None,
            python=None,
            image=None,
            env={},
            secrets={},
            labels=None,
            flavor=None,
            timeout=None,
            namespace=None,
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
            labels=None,
            flavor=None,
            timeout=None,
            namespace=None,
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
            labels=None,
            flavor=None,
            timeout=None,
            namespace=None,
        )
        api.fetch_job_logs.assert_not_called()

    def test_run_fetches_logs_with_correct_namespace(self, runner: CliRunner) -> None:
        """Test that fetch_job_logs uses job.owner.name.

        Regression test for https://github.com/huggingface/huggingface_hub/pull/3736.
        """
        from huggingface_hub._jobs_api import JobOwner

        job_owner = JobOwner(id="user-id", name="my-username", type="user")
        job = Mock(id="my-job-id", owner=job_owner, url="https://huggingface.co/jobs/my-username/my-job-id")
        with (
            patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.jobs._get_extended_environ", return_value={}),
        ):
            api = api_cls.return_value
            api.run_job.return_value = job
            api.fetch_job_logs.return_value = iter(["log line 1"])
            result = runner.invoke(app, ["jobs", "run", "ubuntu", "echo", "hello"])
        assert result.exit_code == 0
        api.fetch_job_logs.assert_called_once_with(job_id="my-job-id", namespace="my-username", follow=True)

    def test_logs_default_no_follow(self, runner: CliRunner) -> None:
        """Test that `hf jobs logs <id>` defaults to follow=False (non-blocking, like `docker logs`)."""
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.fetch_job_logs.return_value = iter(["line 1", "line 2"])
            result = runner.invoke(app, ["jobs", "logs", "my-job-id"])
        assert result.exit_code == 0
        api.fetch_job_logs.assert_called_once_with(job_id="my-job-id", namespace=None, follow=False)
        assert "line 1" in result.output
        assert "line 2" in result.output

    def test_logs_follow_flag(self, runner: CliRunner) -> None:
        """Test that `hf jobs logs -f <id>` passes follow=True."""
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.fetch_job_logs.return_value = iter(["streaming line"])
            result = runner.invoke(app, ["jobs", "logs", "-f", "my-job-id"])
        assert result.exit_code == 0
        api.fetch_job_logs.assert_called_once_with(job_id="my-job-id", namespace=None, follow=True)
        assert "streaming line" in result.output

    def test_logs_follow_long_flag(self, runner: CliRunner) -> None:
        """Test that `hf jobs logs --follow <id>` passes follow=True."""
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.fetch_job_logs.return_value = iter(["streaming line"])
            result = runner.invoke(app, ["jobs", "logs", "--follow", "my-job-id"])
        assert result.exit_code == 0
        api.fetch_job_logs.assert_called_once_with(job_id="my-job-id", namespace=None, follow=True)

    def test_logs_tail(self, runner: CliRunner) -> None:
        """Test that `hf jobs logs --tail 2 <id>` shows only the last 2 lines."""
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.fetch_job_logs.return_value = iter(["line 1", "line 2", "line 3", "line 4"])
            result = runner.invoke(app, ["jobs", "logs", "--tail", "2", "my-job-id"])
        assert result.exit_code == 0
        assert "line 1" not in result.output
        assert "line 2" not in result.output
        assert "line 3" in result.output
        assert "line 4" in result.output

    def test_logs_tail_short_flag(self, runner: CliRunner) -> None:
        """Test that `hf jobs logs -n 1 <id>` shows only the last line."""
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.fetch_job_logs.return_value = iter(["line 1", "line 2", "line 3"])
            result = runner.invoke(app, ["jobs", "logs", "-n", "1", "my-job-id"])
        assert result.exit_code == 0
        assert "line 1" not in result.output
        assert "line 2" not in result.output
        assert "line 3" in result.output

    def test_logs_follow_and_tail_error(self, runner: CliRunner) -> None:
        """Test that `hf jobs logs -f --tail 5 <id>` raises an error."""
        result = runner.invoke(app, ["jobs", "logs", "-f", "--tail", "5", "my-job-id"])
        assert result.exit_code != 0
        assert "Cannot use --follow and --tail together" in str(result.exception)

    def _make_mock_jobs(self):
        """Create mock JobInfo objects for testing ps output."""
        from huggingface_hub._jobs_api import JobInfo

        return [
            JobInfo(
                id="abc123def456",
                createdAt="2026-01-15T10:30:00.000Z",
                dockerImage="python:3.12",
                command=["python", "-c", "print('hello')"],
                arguments=[],
                environment={},
                secrets={},
                flavor="cpu-basic",
                labels={"env": "test"},
                status={"stage": "RUNNING"},
                owner={"id": "user-id", "name": "testuser", "type": "user"},
            ),
            JobInfo(
                id="xyz789ghi012",
                createdAt="2026-01-14T08:00:00.000Z",
                dockerImage="ubuntu:latest",
                command=["echo", "done"],
                arguments=[],
                environment={},
                secrets={},
                flavor="cpu-basic",
                labels={},
                status={"stage": "COMPLETED"},
                owner={"id": "user-id", "name": "testuser", "type": "user"},
            ),
        ]

    def test_ps_format_json(self, runner: CliRunner) -> None:
        """Test that `hf jobs ps -a --format json` outputs valid JSON with all fields."""
        import json

        jobs = self._make_mock_jobs()
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_jobs.return_value = jobs
            result = runner.invoke(app, ["jobs", "ps", "-a", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["id"] == "abc123def456"
        assert data[1]["id"] == "xyz789ghi012"
        # JSON should include all fields, not just table columns
        assert "docker_image" in data[0]
        assert "status" in data[0]
        assert "owner" in data[0]

    def test_ps_json_hidden_alias(self, runner: CliRunner) -> None:
        """Test that `hf jobs ps -a --json` works as alias for `--format json`."""
        import json

        jobs = self._make_mock_jobs()
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_jobs.return_value = jobs
            result = runner.invoke(app, ["jobs", "ps", "-a", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2

    def test_ps_quiet(self, runner: CliRunner) -> None:
        """Test that `hf jobs ps -a -q` outputs only IDs, one per line."""
        jobs = self._make_mock_jobs()
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_jobs.return_value = jobs
            result = runner.invoke(app, ["jobs", "ps", "-a", "-q"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert lines == ["abc123def456", "xyz789ghi012"]

    def test_ps_table_shows_full_ids(self, runner: CliRunner) -> None:
        """Test that table output shows full untruncated job IDs."""
        jobs = self._make_mock_jobs()
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_jobs.return_value = jobs
            result = runner.invoke(app, ["jobs", "ps", "-a"])
        assert result.exit_code == 0
        assert "abc123def456" in result.output
        assert "xyz789ghi012" in result.output

    def test_ps_empty_json(self, runner: CliRunner) -> None:
        """Test that `hf jobs ps --format json` outputs `[]` when no jobs match."""
        import json

        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_jobs.return_value = []
            result = runner.invoke(app, ["jobs", "ps", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    def test_ps_empty_quiet(self, runner: CliRunner) -> None:
        """Test that `hf jobs ps -q` outputs nothing when no jobs match."""
        with patch("huggingface_hub.cli.jobs.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_jobs.return_value = []
            result = runner.invoke(app, ["jobs", "ps", "-q"])
        assert result.exit_code == 0
        assert result.output.strip() == ""
