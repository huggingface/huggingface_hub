# coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for copying files on the Hub.

Covers the [`HfApi.copy_files`] method, the [`CommitOperationCopy`] operation, the
`_resolve_copy_target_path` helper, and the unified `hf cp` CLI command (also exposed
as `hf repos cp` and `hf buckets cp`).
"""

from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from huggingface_hub import HfApi
from huggingface_hub._commit_api import CommitOperationCopy
from huggingface_hub.cli.hf import app
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import _resolve_copy_target_path
from huggingface_hub.utils import SoftTemporaryDirectory

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name


pytestmark = pytest.mark.xet


@pytest.fixture(scope="module")
def api() -> HfApi:
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch):
    """Set HF_TOKEN and HF_ENDPOINT for all CLI tests in this module."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HF_ENDPOINT", ENDPOINT_STAGING)
    yield


# -- Write fixtures (function-scoped, fresh resource per test, cleaned up at the end) --


@pytest.fixture
def repo_write(api: HfApi):
    """Function-scoped model repo for write tests."""
    repo_id = api.create_repo(repo_id=repo_name()).repo_id
    try:
        yield repo_id
    finally:
        api.delete_repo(repo_id=repo_id)


@pytest.fixture
def dataset_write(api: HfApi):
    """Function-scoped dataset repo for write tests."""
    repo_id = api.create_repo(repo_id=repo_name(), repo_type="dataset").repo_id
    try:
        yield repo_id
    finally:
        api.delete_repo(repo_id=repo_id, repo_type="dataset")


@pytest.fixture
def bucket_write(api: HfApi):
    """Function-scoped bucket for write tests."""
    bucket_id = api.create_bucket(repo_name()).bucket_id
    try:
        yield bucket_id
    finally:
        api.delete_bucket(bucket_id)


@pytest.fixture
def bucket_write_2(api: HfApi):
    """Second function-scoped bucket, for bucket-to-bucket / repo-to-bucket write tests."""
    bucket_id = api.create_bucket(repo_name()).bucket_id
    try:
        yield bucket_id
    finally:
        api.delete_bucket(bucket_id)


# -- Read-only fixtures (module-scoped, pre-populated with files, cleaned up at the end) --


@pytest.fixture(scope="module")
def repo_with_files(api: HfApi):
    """Module-scoped model repo pre-populated with files for read-only cp tests."""
    repo_id = api.create_repo(repo_id=repo_name()).repo_id
    api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"hello")
    api.upload_file(repo_id=repo_id, path_in_repo="sub/nested.txt", path_or_fileobj=b"nested content")
    try:
        yield repo_id
    finally:
        api.delete_repo(repo_id=repo_id)


@pytest.fixture(scope="module")
def bucket_with_files(api: HfApi):
    """Module-scoped bucket pre-populated with files for read-only cp tests."""
    bucket_id = api.create_bucket(repo_name()).bucket_id
    api.batch_bucket_files(
        bucket_id,
        add=[
            (b"hello", "file.txt"),
            (b"nested content", "sub/nested.txt"),
        ],
    )
    try:
        yield bucket_id
    finally:
        api.delete_bucket(bucket_id)


# =============================================================================
# CommitOperationCopy (unit tests, no network)
# =============================================================================


class TestCommitOperationCopy:
    def test_cross_repo_copy_missing_repo_id_or_type(self):
        with pytest.raises(ValueError, match="`src_repo_type` is required when `src_repo_id` is set"):
            CommitOperationCopy(src_path_in_repo="src.bin", path_in_repo="dst.bin", src_repo_id="user/source")

        with pytest.raises(ValueError, match="`src_repo_id` is required when `src_repo_type` is set"):
            CommitOperationCopy(src_path_in_repo="src.bin", path_in_repo="dst.bin", src_repo_type="model")

    def test_path_normalization(self):
        op = CommitOperationCopy(src_path_in_repo="./src.bin", path_in_repo="/dst.bin")
        assert op.src_path_in_repo == "src.bin"
        assert op.path_in_repo == "dst.bin"


# =============================================================================
# _resolve_copy_target_path (unit tests, no network)
# =============================================================================


_RESOLVE_DEFAULTS = {
    "src_file_path": "file.txt",
    "src_root_path": None,
    "is_single_file": True,
    "destination_path": "",
    "destination_is_directory": False,
    "destination_exists_as_directory": False,
    "merge_contents": False,
}


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # Single file cases
        ({"src_file_path": "file.txt", "destination_path": ""}, "file.txt"),
        ({"src_file_path": "file.txt", "destination_path": "renamed.txt"}, "renamed.txt"),
        ({"src_file_path": "file.txt", "destination_path": "dir", "destination_is_directory": True}, "dir/file.txt"),
        # Folder to nonexistent destination (rename semantics)
        (
            {
                "src_file_path": "folder/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "target",
            },
            "target/a.txt",
        ),
        # Folder to existing directory (cp -r nesting)
        (
            {
                "src_file_path": "folder/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "target",
                "destination_exists_as_directory": True,
            },
            "target/folder/a.txt",
        ),
        # Trailing slash on source (rsync semantics, no nesting)
        (
            {
                "src_file_path": "folder/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "target",
                "destination_exists_as_directory": True,
                "merge_contents": True,
            },
            "target/a.txt",
        ),
        # Folder to root (existing dir)
        (
            {
                "src_file_path": "folder/sub/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "",
                "destination_exists_as_directory": True,
            },
            "folder/sub/a.txt",
        ),
        # Folder contents to root
        (
            {
                "src_file_path": "folder/sub/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "",
                "destination_exists_as_directory": True,
                "merge_contents": True,
            },
            "sub/a.txt",
        ),
        # Nested subfolder
        (
            {
                "src_file_path": "data/train/a.csv",
                "src_root_path": "data",
                "is_single_file": False,
                "destination_path": "backup",
            },
            "backup/train/a.csv",
        ),
    ],
)
def test_resolve_copy_target_path(kwargs, expected):
    assert _resolve_copy_target_path(**{**_RESOLVE_DEFAULTS, **kwargs}) == expected


# =============================================================================
# copy_files / CommitOperationCopy (against staging)
# =============================================================================


def test_commit_copy_file(api: HfApi, repo_write: str) -> None:
    """Test CommitOperationCopy.

    Works only when copying an LFS file.
    """
    api.upload_file(path_or_fileobj=b"content", repo_id=repo_write, path_in_repo="file.txt")
    api.upload_file(path_or_fileobj=b"LFS content", repo_id=repo_write, path_in_repo="lfs.bin")

    api.create_commit(
        repo_id=repo_write,
        commit_message="Copy LFS file.",
        operations=[
            CommitOperationCopy(src_path_in_repo="lfs.bin", path_in_repo="lfs Copy.bin"),
            CommitOperationCopy(src_path_in_repo="lfs.bin", path_in_repo="lfs Copy (1).bin"),
        ],
    )
    api.create_commit(
        repo_id=repo_write,
        commit_message="Copy regular file.",
        operations=[CommitOperationCopy(src_path_in_repo="file.txt", path_in_repo="file Copy.txt")],
    )
    with pytest.raises(EntryNotFoundError):
        api.create_commit(
            repo_id=repo_write,
            commit_message="Copy a file that doesn't exist.",
            operations=[
                CommitOperationCopy(src_path_in_repo="doesnt-exist.txt", path_in_repo="doesnt-exist Copy.txt")
            ],
        )

    repo_files = api.list_repo_files(repo_id=repo_write)
    assert "file.txt" in repo_files
    assert "file Copy.txt" in repo_files
    assert "lfs.bin" in repo_files
    assert "lfs Copy.bin" in repo_files
    assert "lfs Copy (1).bin" in repo_files

    repo_file1, repo_file2 = api.get_paths_info(repo_id=repo_write, paths=["lfs.bin", "lfs Copy.bin"])
    assert repo_file1.lfs["sha256"] == repo_file2.lfs["sha256"]


def test_copy_files_repo_to_repo(api: HfApi, repo_write: str, dataset_write: str):
    """Test copy from model repo to dataset repo"""
    dst_repo_id = dataset_write

    api.upload_file(repo_id=repo_write, path_in_repo="config.json", path_or_fileobj=b'{"key": "value"}')
    api.upload_file(repo_id=repo_write, path_in_repo="data/a.txt", path_or_fileobj=b"text data")
    api.upload_file(repo_id=repo_write, path_in_repo="data/sub/lfs.bin", path_or_fileobj=b"binary data")

    # Copy a single file
    api.copy_files(f"hf://{repo_write}/config.json", f"hf://datasets/{dst_repo_id}/config.json")

    # Copy folder (1 text, 1 LFS)
    api.copy_files(f"hf://{repo_write}/data/", f"hf://datasets/{dst_repo_id}/copied/")

    # Verify files were copied
    dst_files = api.list_repo_files(dst_repo_id, repo_type="dataset")
    assert "config.json" in dst_files
    assert "copied/a.txt" in dst_files
    assert "copied/sub/lfs.bin" in dst_files

    # Check data is valid
    with SoftTemporaryDirectory() as tmpdir:
        copied_config_path = Path(
            api.hf_hub_download(dst_repo_id, "config.json", cache_dir=tmpdir, repo_type="dataset")
        )
        copied_txt_path = Path(api.hf_hub_download(dst_repo_id, "copied/a.txt", cache_dir=tmpdir, repo_type="dataset"))
        copied_bin_path = Path(
            api.hf_hub_download(dst_repo_id, "copied/sub/lfs.bin", cache_dir=tmpdir, repo_type="dataset")
        )

        assert copied_config_path.read_bytes() == b'{"key": "value"}'
        assert copied_txt_path.read_bytes() == b"text data"
        assert copied_bin_path.read_bytes() == b"binary data"


# =============================================================================
# CLI `hf cp` (against staging)
# =============================================================================


def cli(command: str, input: str | None = None) -> Result:
    """Invoke a CLI command, e.g. `cli("hf cp a.txt hf://buckets/u/b")`."""
    assert command.startswith("hf ")
    args = command.split(" ")[1:]
    return CliRunner().invoke(app, [*args], input=input)


def _remote_files(api: HfApi, bucket_id: str) -> set[str]:
    """Return set of file paths in a bucket."""
    return {f.path for f in api.list_bucket_tree(bucket_id)}


# -- Upload tests (local -> bucket) --


def test_cp_upload_file_to_bucket_root(api: HfApi, bucket_write: str, tmp_path: Path):
    """Upload a local file to a bucket (no prefix)."""
    local_file = tmp_path / "local.txt"
    local_file.write_text("upload me")

    result = cli(f"hf cp {local_file} hf://buckets/{bucket_write}")
    assert result.exit_code == 0
    assert "Uploaded" in result.output

    # Verify file exists in bucket with basename as remote path
    assert "local.txt" in _remote_files(api, bucket_write)


def test_cp_upload_file_to_bucket_prefix(api: HfApi, bucket_write: str, tmp_path: Path):
    """Upload a local file to a bucket subdirectory (trailing slash prefix)."""
    local_file = tmp_path / "data.csv"
    local_file.write_text("a,b,c")

    result = cli(f"hf cp {local_file} hf://buckets/{bucket_write}/logs/")
    assert result.exit_code == 0

    assert "logs/data.csv" in _remote_files(api, bucket_write)


def test_cp_upload_file_with_remote_name(api: HfApi, bucket_write: str, tmp_path: Path):
    """Upload a local file with an explicit remote filename."""
    local_file = tmp_path / "original.txt"
    local_file.write_text("renamed upload")

    result = cli(f"hf cp {local_file} hf://buckets/{bucket_write}/remote-name.txt")
    assert result.exit_code == 0

    assert "remote-name.txt" in _remote_files(api, bucket_write)


def test_cp_upload_file_quiet(api: HfApi, bucket_write: str, tmp_path: Path):
    """Upload with --quiet suppresses output."""
    local_file = tmp_path / "quiet.txt"
    local_file.write_text("quiet")

    result = cli(f"hf cp {local_file} hf://buckets/{bucket_write}/quiet.txt --quiet")
    assert result.exit_code == 0
    assert "Uploaded:" not in result.output


def test_cp_upload_from_stdin(bucket_write: str):
    """Upload file content from stdin."""
    result = cli(f"hf cp - hf://buckets/{bucket_write}/from-stdin.txt", input="stdin data")
    assert result.exit_code == 0
    assert "Uploaded" in result.output


# -- Download tests (bucket -> local) --


def test_cp_download_to_explicit_file(bucket_with_files: str, tmp_path: Path):
    """Download a bucket file to a specific local path."""
    output_file = tmp_path / "output.txt"
    result = cli(f"hf cp hf://buckets/{bucket_with_files}/file.txt {output_file}")
    assert result.exit_code == 0
    assert "Downloaded" in result.output
    assert output_file.read_text() == "hello"


def test_cp_download_to_directory(bucket_with_files: str, tmp_path: Path):
    """Download a bucket file to an existing directory (uses original filename)."""
    result = cli(f"hf cp hf://buckets/{bucket_with_files}/file.txt {tmp_path}/")
    assert result.exit_code == 0

    downloaded = tmp_path / "file.txt"
    assert downloaded.exists()
    assert downloaded.read_text() == "hello"


def test_cp_download_nested_file(bucket_with_files: str, tmp_path: Path):
    """Download a file from a subdirectory in the bucket."""
    output_file = tmp_path / "nested.txt"
    result = cli(f"hf cp hf://buckets/{bucket_with_files}/sub/nested.txt {output_file}")
    assert result.exit_code == 0
    assert output_file.read_text() == "nested content"


def test_cp_download_to_stdout(bucket_with_files: str):
    """Download a bucket file to stdout."""
    result = cli(f"hf cp hf://buckets/{bucket_with_files}/file.txt -")
    assert result.exit_code == 0
    assert "hello" in result.output
    # stdout mode should NOT print "Downloaded:" message
    assert "Downloaded:" not in result.output


def test_cp_download_quiet(bucket_with_files: str, tmp_path: Path):
    """Download with --quiet suppresses the status message."""
    output_file = tmp_path / "quiet-download.txt"
    result = cli(f"hf cp hf://buckets/{bucket_with_files}/file.txt {output_file} --quiet")
    assert result.exit_code == 0
    assert "Downloaded:" not in result.output
    assert output_file.read_text() == "hello"


def test_cp_download_creates_parent_dirs(bucket_with_files: str, tmp_path: Path):
    """Download creates parent directories when they don't exist."""
    output_file = tmp_path / "a" / "b" / "c" / "output.txt"
    result = cli(f"hf cp hf://buckets/{bucket_with_files}/file.txt {output_file}")
    assert result.exit_code == 0
    assert output_file.read_text() == "hello"


# -- Repo upload / download tests (local <-> repo) --


def test_cp_upload_file_to_repo(api: HfApi, repo_write: str, tmp_path: Path):
    """Upload a local file to a repository."""
    local_file = tmp_path / "weights.bin"
    local_file.write_text("model weights")

    result = cli(f"hf cp {local_file} hf://{repo_write}/models/weights.bin")
    assert result.exit_code == 0
    assert "Uploaded" in result.output

    assert "models/weights.bin" in api.list_repo_files(repo_write)


def test_cp_upload_to_repo_prefix(api: HfApi, repo_write: str, tmp_path: Path):
    """Upload a local file to a repo subdirectory (basename appended on trailing slash)."""
    local_file = tmp_path / "data.csv"
    local_file.write_text("a,b,c")

    result = cli(f"hf cp {local_file} hf://{repo_write}/logs/")
    assert result.exit_code == 0

    assert "logs/data.csv" in api.list_repo_files(repo_write)


def test_cp_upload_to_repo_from_stdin(api: HfApi, repo_write: str):
    """Upload file content from stdin to a repo."""
    result = cli(f"hf cp - hf://{repo_write}/from-stdin.txt", input="stdin data")
    assert result.exit_code == 0
    assert "Uploaded" in result.output

    assert "from-stdin.txt" in api.list_repo_files(repo_write)


def test_cp_download_repo_file_to_local(repo_with_files: str, tmp_path: Path):
    """Download a repo file to a specific local path."""
    output_file = tmp_path / "output.txt"
    result = cli(f"hf cp hf://{repo_with_files}/file.txt {output_file}")
    assert result.exit_code == 0
    assert "Downloaded" in result.output
    assert output_file.read_text() == "hello"


def test_cp_download_repo_nested_file_to_directory(repo_with_files: str, tmp_path: Path):
    """Download a nested repo file to a directory (uses original filename)."""
    result = cli(f"hf cp hf://{repo_with_files}/sub/nested.txt {tmp_path}/")
    assert result.exit_code == 0

    downloaded = tmp_path / "nested.txt"
    assert downloaded.exists()
    assert downloaded.read_text() == "nested content"


def test_cp_download_repo_file_to_stdout(repo_with_files: str):
    """Download a repo file to stdout."""
    result = cli(f"hf cp hf://{repo_with_files}/file.txt -")
    assert result.exit_code == 0
    assert "hello" in result.output
    assert "Downloaded:" not in result.output


def test_cp_download_repo_file_from_revision(api: HfApi, repo_write: str):
    """Download a repo file pinned to a specific branch via `@revision`."""
    branch = "cp-branch"
    api.create_branch(repo_id=repo_write, branch=branch)
    api.upload_file(repo_id=repo_write, path_in_repo="branch.txt", path_or_fileobj=b"from branch", revision=branch)

    result = cli(f"hf cp hf://{repo_write}@{branch}/branch.txt -")
    assert result.exit_code == 0
    assert "from branch" in result.output


# -- Remote to remote tests --


def test_cp_remote_repo_to_repo(api: HfApi, repo_write: str, dataset_write: str):
    """Copy a single file and a folder from one repo to another via the CLI."""
    dst_repo_id = dataset_write

    api.upload_file(repo_id=repo_write, path_in_repo="config.json", path_or_fileobj=b'{"key": "value"}')
    api.upload_file(repo_id=repo_write, path_in_repo="data/a.txt", path_or_fileobj=b"text data")

    cli(f"hf cp hf://{repo_write}/config.json hf://datasets/{dst_repo_id}/config.json")
    cli(f"hf cp hf://{repo_write}/data/ hf://datasets/{dst_repo_id}/copied/")

    dst_files = api.list_repo_files(dst_repo_id, repo_type="dataset")
    assert "config.json" in dst_files
    assert "copied/a.txt" in dst_files


def test_cp_remote_bucket_to_bucket(api: HfApi, bucket_write: str, bucket_write_2: str):
    api.batch_bucket_files(bucket_write, add=[(b"aaa", "logs/a.txt"), (b"bbb", "logs/sub/b.txt"), (b"ccc", "c.txt")])

    cli(f"hf cp hf://buckets/{bucket_write}/logs hf://buckets/{bucket_write_2}/backup/")

    files = _remote_files(api, bucket_write_2)
    assert "backup/a.txt" in files
    assert "backup/sub/b.txt" in files
    assert "backup/c.txt" not in files


def test_cp_remote_repo_to_bucket(api: HfApi, repo_write: str, bucket_write_2: str):
    branch = "cp-copy-branch"

    api.upload_file(repo_id=repo_write, path_in_repo="main.txt", path_or_fileobj=b"main")
    api.create_branch(repo_id=repo_write, branch=branch)
    api.upload_file(
        repo_id=repo_write, path_in_repo="nested/from-branch.txt", path_or_fileobj=b"branch", revision=branch
    )

    cli(f"hf cp hf://{repo_write}@{branch}/nested/from-branch.txt hf://buckets/{bucket_write_2}/copied.txt")

    assert "copied.txt" in _remote_files(api, bucket_write_2)


def test_cp_error_bucket_to_repo():
    result = cli("hf cp hf://buckets/username/bucket-name/file.txt hf://username/repo-name/file.txt")
    assert result.exit_code != 0
    assert str(result.exception) == "Bucket-to-repo copy is not supported."


# -- Validation errors (no network) --


def test_cp_error_both_local(tmp_path: Path):
    """Both src and dst are local paths."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    result = cli(f"hf cp {src} {dst}")
    assert result.exit_code != 0
    assert "must be a repo" in result.output.lower()


def test_cp_error_missing_destination(tmp_path: Path):
    """Local src without a destination."""
    src = tmp_path / "orphan.txt"
    result = cli(f"hf cp {src}")
    assert result.exit_code != 0
    assert "Missing destination" in result.output


@pytest.mark.parametrize(
    "cmd",
    [
        # Cannot copy to a bucket
        f"hf repos cp hf://{USER}/some-model/c.json hf://buckets/{USER}/some-bucket/c.json",
        # Cannot download from a bucket
        f"hf repos cp hf://buckets/{USER}/some-bucket/c.json ./c.json",
    ],
)
def test_cp_error_repos_alias_rejects_bucket_remote(cmd: str):
    result = cli(cmd)
    assert result.exit_code != 0
    assert "`hf repos cp` only works with repositories" in str(result.exception)


@pytest.mark.parametrize(
    "cmd",
    [
        # Cannot copy to a repo
        f"hf buckets cp hf://buckets/{USER}/some-bucket/c.json hf://{USER}/some-model/c.json",
        # Cannot download from a repo
        f"hf buckets cp hf://{USER}/some-model/c.json -",
    ],
)
def test_cp_error_buckets_alias_rejects_repo_remote(cmd: str):
    result = cli(cmd)
    assert result.exit_code != 0
    assert "`hf buckets cp` only works with buckets" in str(result.exception)


def test_cp_error_stdin_to_local():
    """Stdin upload to a local path is not allowed (must target a repo or bucket)."""
    result = cli("hf cp - /tmp/local.txt", input="data")
    assert result.exit_code != 0
    assert "must be a repo" in result.output.lower()


def test_cp_error_stdin_no_filename_empty_prefix():
    """Stdin upload to a remote path without a filename (empty prefix)."""
    result = cli(f"hf cp - hf://buckets/{USER}/some-bucket", input="data")
    assert result.exit_code != 0
    assert "full destination path including filename" in result.output


def test_cp_error_stdin_no_filename_trailing_slash():
    """Stdin upload to a remote path with trailing slash (no filename)."""
    result = cli(f"hf cp - hf://buckets/{USER}/some-bucket/logs/", input="data")
    assert result.exit_code != 0
    assert "full destination path including filename" in result.output


def test_cp_error_stdout_with_local_source(tmp_path: Path):
    """Cannot pipe to stdout when source is not a remote URI."""
    src = tmp_path / "local.txt"
    result = cli(f"hf cp {src} -")
    assert result.exit_code != 0
    assert "must be a repo" in result.output.lower()


def test_cp_error_source_is_directory(tmp_path: Path):
    """Source must be a file, not a directory."""
    result = cli(f"hf cp {tmp_path} hf://buckets/{USER}/some-bucket/file.txt")
    assert result.exit_code != 0
    assert "source must be a file, not a directory." in result.output.lower()


def test_cp_error_source_file_not_found():
    """Source file does not exist."""
    result = cli(f"hf cp /nonexistent/file.txt hf://buckets/{USER}/some-bucket/file.txt")
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_cp_error_remote_source_not_found(bucket_with_files: str):
    """Copying a non-existent file from a bucket raises an error."""
    result = cli(f"hf cp hf://buckets/{bucket_with_files}/doesnotexist.txt hf://buckets/{bucket_with_files}/out.txt")
    assert result.exit_code != 0
    assert isinstance(result.exception, EntryNotFoundError)
