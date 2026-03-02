"""Tests for `hf repos list` CLI command."""

import json
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from huggingface_hub.cli.hf import app
from huggingface_hub.cli.repos import _parse_repo_argument
from huggingface_hub.hf_api import ModelInfo, RepoFile, RepoFolder
from huggingface_hub.utils._hf_url import HfUrl, parse_hf_url


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_model_info(id: str, **kwargs) -> ModelInfo:
    return ModelInfo(id=id, **kwargs)


# ---------------------------------------------------------------------------
# parse_hf_url unit tests (shared utility)
# ---------------------------------------------------------------------------


class TestParseHfUrl:
    def test_empty(self):
        assert parse_hf_url("") == HfUrl()

    def test_models_only(self):
        assert parse_hf_url("hf://models") == HfUrl(resource_type="model")

    def test_datasets_only(self):
        assert parse_hf_url("hf://datasets") == HfUrl(resource_type="dataset")

    def test_spaces_only(self):
        assert parse_hf_url("hf://spaces") == HfUrl(resource_type="space")

    def test_buckets_only(self):
        assert parse_hf_url("hf://buckets") == HfUrl(resource_type="bucket")

    def test_models_namespace(self):
        assert parse_hf_url("hf://models/huggingface") == HfUrl(resource_type="model", repo_id="huggingface")

    def test_datasets_repo(self):
        assert parse_hf_url("hf://datasets/user/repo") == HfUrl(resource_type="dataset", repo_id="user/repo")

    def test_repo_with_revision(self):
        assert parse_hf_url("hf://datasets/user/repo@main") == HfUrl(
            resource_type="dataset", repo_id="user/repo", revision="main"
        )

    def test_repo_with_revision_and_path(self):
        assert parse_hf_url("hf://datasets/user/repo@v1/data/train") == HfUrl(
            resource_type="dataset", repo_id="user/repo", revision="v1", path="data/train"
        )

    def test_bucket_with_path(self):
        assert parse_hf_url("hf://buckets/user/bucket/prefix/file.txt") == HfUrl(
            resource_type="bucket", repo_id="user/bucket", path="prefix/file.txt"
        )

    def test_bucket_no_prefix(self):
        assert parse_hf_url("hf://buckets/user/bucket") == HfUrl(resource_type="bucket", repo_id="user/bucket")

    def test_plain_repo_id(self):
        assert parse_hf_url("user/repo") == HfUrl(repo_id="user/repo")

    def test_plain_namespace(self):
        assert parse_hf_url("namespace") == HfUrl(repo_id="namespace")

    def test_plain_repo_with_path(self):
        assert parse_hf_url("user/repo/path/to/file") == HfUrl(repo_id="user/repo", path="path/to/file")

    def test_plain_repo_with_revision(self):
        assert parse_hf_url("user/repo@dev") == HfUrl(repo_id="user/repo", revision="dev")

    def test_special_refs_pr(self):
        assert parse_hf_url("user/repo@refs/pr/123/some/path") == HfUrl(
            repo_id="user/repo", revision="refs/pr/123", path="some/path"
        )

    def test_special_refs_convert(self):
        assert parse_hf_url("user/repo@refs/convert/parquet") == HfUrl(
            repo_id="user/repo", revision="refs/convert/parquet"
        )

    def test_url_encoded_revision(self):
        assert parse_hf_url("user/repo@my%20branch") == HfUrl(repo_id="user/repo", revision="my branch")

    def test_no_hf_prefix_with_type(self):
        assert parse_hf_url("models/user/repo") == HfUrl(resource_type="model", repo_id="user/repo")

    def test_hf_prefix_default_type(self):
        assert parse_hf_url("hf://user/repo") == HfUrl(repo_id="user/repo")

    def test_single_word_with_revision(self):
        assert parse_hf_url("gpt2@dev") == HfUrl(repo_id="gpt2", revision="dev")


# ---------------------------------------------------------------------------
# _parse_repo_argument unit tests
# ---------------------------------------------------------------------------


class TestParseRepoArgument:
    def test_empty_string(self):
        assert _parse_repo_argument("") == ("model", None, None, None)

    def test_namespace_only(self):
        assert _parse_repo_argument("huggingface") == ("model", "huggingface", None, None)

    def test_repo_id(self):
        assert _parse_repo_argument("user/my-model") == ("model", "user/my-model", None, "")

    def test_repo_id_with_path(self):
        assert _parse_repo_argument("user/my-model/sub/folder") == ("model", "user/my-model", None, "sub/folder")

    def test_repo_id_with_revision(self):
        assert _parse_repo_argument("user/my-model@main") == ("model", "user/my-model", "main", "")

    def test_repo_id_with_revision_and_path(self):
        assert _parse_repo_argument("user/my-model@v1.0/data/train") == (
            "model",
            "user/my-model",
            "v1.0",
            "data/train",
        )

    def test_hf_models_only(self):
        assert _parse_repo_argument("hf://models") == ("model", None, None, None)

    def test_hf_models_namespace(self):
        assert _parse_repo_argument("hf://models/huggingface") == ("model", "huggingface", None, None)

    def test_hf_datasets_only(self):
        assert _parse_repo_argument("hf://datasets") == ("dataset", None, None, None)

    def test_hf_datasets_namespace(self):
        assert _parse_repo_argument("hf://datasets/huggingface") == ("dataset", "huggingface", None, None)

    def test_hf_spaces_repo(self):
        assert _parse_repo_argument("hf://spaces/user/my-space") == ("space", "user/my-space", None, "")

    def test_hf_datasets_repo(self):
        assert _parse_repo_argument("hf://datasets/user/my-dataset") == ("dataset", "user/my-dataset", None, "")

    def test_hf_datasets_repo_with_revision(self):
        assert _parse_repo_argument("hf://datasets/user/my-dataset@main") == (
            "dataset",
            "user/my-dataset",
            "main",
            "",
        )

    def test_hf_datasets_repo_with_revision_and_path(self):
        assert _parse_repo_argument("hf://datasets/user/my-dataset@v1/train") == (
            "dataset",
            "user/my-dataset",
            "v1",
            "train",
        )

    def test_hf_models_repo_with_path(self):
        assert _parse_repo_argument("hf://models/user/my-model/checkpoints") == (
            "model",
            "user/my-model",
            None,
            "checkpoints",
        )

    def test_hf_default_type_is_model(self):
        assert _parse_repo_argument("hf://user/my-model") == ("model", "user/my-model", None, "")

    def test_hf_user_model_with_revision(self):
        assert _parse_repo_argument("hf://user/my-model@patch-ref") == (
            "model",
            "user/my-model",
            "patch-ref",
            "",
        )

    def test_hf_user_model_with_revision_and_path(self):
        assert _parse_repo_argument("hf://user/my-model@revision/path/to/folder") == (
            "model",
            "user/my-model",
            "revision",
            "path/to/folder",
        )

    def test_repo_type_from_handle_overrides(self):
        # If handle says dataset, repo_type=None should pick that up
        assert _parse_repo_argument("hf://datasets/user/repo", repo_type=None) == (
            "dataset",
            "user/repo",
            None,
            "",
        )

    def test_repo_type_conflict_raises(self):
        with pytest.raises(ValueError, match="conflicts"):
            _parse_repo_argument("hf://datasets/user/repo", repo_type="model")

    def test_repo_type_passed_explicitly(self):
        assert _parse_repo_argument("user/repo", repo_type="dataset") == ("dataset", "user/repo", None, "")

    def test_special_refs_pr(self):
        assert _parse_repo_argument("user/repo@refs/pr/123/some/path") == (
            "model",
            "user/repo",
            "refs/pr/123",
            "some/path",
        )

    def test_special_refs_convert(self):
        assert _parse_repo_argument("user/repo@refs/convert/parquet") == (
            "model",
            "user/repo",
            "refs/convert/parquet",
            "",
        )

    def test_url_encoded_revision(self):
        assert _parse_repo_argument("user/repo@my%20branch") == (
            "model",
            "user/repo",
            "my branch",
            "",
        )

    def test_single_word_with_revision(self):
        assert _parse_repo_argument("repo-name@v1") == ("model", "repo-name", "v1", "")


# ---------------------------------------------------------------------------
# Helper to build mock RepoFile / RepoFolder
# ---------------------------------------------------------------------------


def _make_repo_file(path: str, size: int = 100, date: Optional[datetime] = None) -> RepoFile:
    f = RepoFile(path=path, size=size, oid="abc123")
    if date:
        from huggingface_hub.hf_api import LastCommitInfo

        f.last_commit = LastCommitInfo(oid="abc", title="test", date=date)
    return f


def _make_repo_folder(path: str, date: Optional[datetime] = None) -> RepoFolder:
    f = RepoFolder(path=path, oid="def456")
    if date:
        from huggingface_hub.hf_api import LastCommitInfo

        f.last_commit = LastCommitInfo(oid="abc", title="test", date=date)
    return f


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestReposListRepos:
    """Tests for listing repos (no repo_id provided)."""

    def test_list_models_default(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = [_make_model_info("user/my-model")]
            result = runner.invoke(app, ["repos", "list"])

        assert result.exit_code == 0
        api.list_models.assert_called_once_with(author=None)

    def test_list_models_with_namespace(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = []
            api.whoami.return_value = {"name": "test-user"}
            result = runner.invoke(app, ["repos", "list", "huggingface"])

        assert result.exit_code == 0
        api.list_models.assert_called_once_with(author="huggingface")

    def test_list_datasets_with_type(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_datasets.return_value = []
            api.whoami.return_value = {"name": "test-user"}
            result = runner.invoke(app, ["repos", "list", "--type", "dataset"])

        assert result.exit_code == 0
        api.list_datasets.assert_called_once_with(author=None)

    def test_list_spaces_with_type(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_spaces.return_value = []
            api.whoami.return_value = {"name": "test-user"}
            result = runner.invoke(app, ["repos", "list", "--type", "space"])

        assert result.exit_code == 0
        api.list_spaces.assert_called_once_with(author=None)

    def test_list_datasets_via_handle(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_datasets.return_value = []
            api.whoami.return_value = {"name": "test-user"}
            result = runner.invoke(app, ["repos", "list", "hf://datasets"])

        assert result.exit_code == 0
        api.list_datasets.assert_called_once_with(author=None)

    def test_list_models_via_handle_with_namespace(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = []
            api.whoami.return_value = {"name": "test-user"}
            result = runner.invoke(app, ["repos", "list", "hf://models/meta-llama"])

        assert result.exit_code == 0
        api.list_models.assert_called_once_with(author="meta-llama")

    def test_list_repos_no_results(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = []
            api.whoami.return_value = {"name": "test-user"}
            result = runner.invoke(app, ["repos", "list"])

        assert result.exit_code == 0
        assert "No models found" in result.stdout

    def test_list_repos_quiet(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = [_make_model_info("user/my-model")]
            result = runner.invoke(app, ["repos", "list", "-q"])

        assert result.exit_code == 0
        assert "user/my-model" in result.stdout

    def test_list_repos_json(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = [_make_model_info("user/my-model")]
            result = runner.invoke(app, ["repos", "list", "--format", "json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_tree_rejected_for_repos(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = []
            result = runner.invoke(app, ["repos", "list", "--tree"])

        assert result.exit_code != 0

    def test_recursive_rejected_for_repos(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_models.return_value = []
            result = runner.invoke(app, ["repos", "list", "-R"])

        assert result.exit_code != 0


class TestReposListFiles:
    """Tests for listing files in a repo (repo_id provided)."""

    def test_list_files_basic(self, runner: CliRunner) -> None:
        items = [
            _make_repo_file("config.json", size=200),
            _make_repo_file("model.safetensors", size=1000000),
            _make_repo_folder("data"),
        ]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model"])

        assert result.exit_code == 0
        assert "config.json" in result.stdout
        assert "model.safetensors" in result.stdout
        assert "data/" in result.stdout
        api.list_repo_tree.assert_called_once_with(
            "user/my-model",
            path_in_repo=None,
            recursive=False,
            expand=True,
            revision=None,
            repo_type="model",
        )

    def test_list_files_recursive(self, runner: CliRunner) -> None:
        items = [_make_repo_file("config.json")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model", "-R"])

        assert result.exit_code == 0
        api.list_repo_tree.assert_called_once_with(
            "user/my-model",
            path_in_repo=None,
            recursive=True,
            expand=True,
            revision=None,
            repo_type="model",
        )

    def test_list_files_with_path(self, runner: CliRunner) -> None:
        items = [_make_repo_file("data/train.csv")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model/data", "-R"])

        assert result.exit_code == 0
        api.list_repo_tree.assert_called_once_with(
            "user/my-model",
            path_in_repo="data",
            recursive=True,
            expand=True,
            revision=None,
            repo_type="model",
        )

    def test_list_files_with_revision(self, runner: CliRunner) -> None:
        items = [_make_repo_file("config.json")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model@dev"])

        assert result.exit_code == 0
        api.list_repo_tree.assert_called_once_with(
            "user/my-model",
            path_in_repo=None,
            recursive=False,
            expand=True,
            revision="dev",
            repo_type="model",
        )

    def test_list_files_hf_handle_dataset(self, runner: CliRunner) -> None:
        items = [_make_repo_file("train.csv")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "hf://datasets/user/my-dataset"])

        assert result.exit_code == 0
        api.list_repo_tree.assert_called_once_with(
            "user/my-dataset",
            path_in_repo=None,
            recursive=False,
            expand=True,
            revision=None,
            repo_type="dataset",
        )

    def test_list_files_hf_handle_with_revision(self, runner: CliRunner) -> None:
        items = [_make_repo_file("config.json")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "hf://datasets/user/my-dataset@patch-ref"])

        assert result.exit_code == 0
        api.list_repo_tree.assert_called_once_with(
            "user/my-dataset",
            path_in_repo=None,
            recursive=False,
            expand=True,
            revision="patch-ref",
            repo_type="dataset",
        )

    def test_list_files_quiet(self, runner: CliRunner) -> None:
        items = [
            _make_repo_file("config.json"),
            _make_repo_folder("data"),
        ]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model", "-q"])

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        assert "config.json" in lines
        assert "data/" in lines

    def test_list_files_json(self, runner: CliRunner) -> None:
        items = [_make_repo_file("config.json", size=512)]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model", "--format", "json"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["path"] == "config.json"
        assert data[0]["size"] == 512

    def test_list_files_tree(self, runner: CliRunner) -> None:
        items = [
            _make_repo_file("config.json"),
            _make_repo_file("data/train.csv"),
            _make_repo_folder("data"),
        ]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model", "--tree"])

        assert result.exit_code == 0
        assert "├── " in result.stdout or "└── " in result.stdout

    def test_list_files_tree_with_human_readable(self, runner: CliRunner) -> None:
        dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        items = [
            _make_repo_file("config.json", size=512, date=dt),
            _make_repo_file("model.safetensors", size=2_500_000, date=dt),
        ]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model", "--tree", "-h"])

        assert result.exit_code == 0
        assert "512 B" in result.stdout
        assert "2.5 MB" in result.stdout

    def test_list_files_human_readable(self, runner: CliRunner) -> None:
        dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        items = [
            _make_repo_file("model.safetensors", size=2_500_000, date=dt),
        ]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model", "-h"])

        assert result.exit_code == 0
        assert "2.5 MB" in result.stdout

    def test_list_files_empty(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = []
            result = runner.invoke(app, ["repos", "list", "user/my-model"])

        assert result.exit_code == 0
        assert "(empty)" in result.stdout

    def test_list_files_tree_json_rejected(self, runner: CliRunner) -> None:
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = []
            result = runner.invoke(app, ["repos", "list", "user/my-model", "--tree", "--format", "json"])

        assert result.exit_code != 0

    def test_list_files_shows_hint_for_directories(self, runner: CliRunner) -> None:
        items = [
            _make_repo_file("config.json"),
            _make_repo_folder("data"),
        ]
        with (
            patch("huggingface_hub.cli.repos.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.repos.StatusLine") as mock_status,
        ):
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model"])

        assert result.exit_code == 0
        mock_status.return_value.done.assert_called_once_with("Use -R to list files recursively.")

    def test_list_files_no_hint_when_recursive(self, runner: CliRunner) -> None:
        items = [
            _make_repo_file("config.json"),
            _make_repo_folder("data"),
        ]
        with (
            patch("huggingface_hub.cli.repos.get_hf_api") as api_cls,
            patch("huggingface_hub.cli.repos.StatusLine") as mock_status,
        ):
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-model", "-R"])

        assert result.exit_code == 0
        mock_status.return_value.done.assert_not_called()

    def test_ls_alias(self, runner: CliRunner) -> None:
        items = [_make_repo_file("config.json")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "ls", "user/my-model"])

        assert result.exit_code == 0
        assert "config.json" in result.stdout

    def test_list_files_dataset_with_type_flag(self, runner: CliRunner) -> None:
        items = [_make_repo_file("train.csv")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "user/my-dataset", "--type", "dataset"])

        assert result.exit_code == 0
        api.list_repo_tree.assert_called_once_with(
            "user/my-dataset",
            path_in_repo=None,
            recursive=False,
            expand=True,
            revision=None,
            repo_type="dataset",
        )

    def test_list_files_hf_handle_with_revision_and_path(self, runner: CliRunner) -> None:
        items = [_make_repo_file("train.csv")]
        with patch("huggingface_hub.cli.repos.get_hf_api") as api_cls:
            api = api_cls.return_value
            api.list_repo_tree.return_value = items
            result = runner.invoke(app, ["repos", "list", "hf://datasets/user/my-dataset@main/data"])

        assert result.exit_code == 0
        api.list_repo_tree.assert_called_once_with(
            "user/my-dataset",
            path_in_repo="data",
            recursive=False,
            expand=True,
            revision="main",
            repo_type="dataset",
        )
