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
import json
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner, Result

from huggingface_hub import HfApi
from huggingface_hub.cli.hf import app
from huggingface_hub.errors import BucketNotFoundError, HfHubHTTPError
from huggingface_hub.hf_api import _split_bucket_id_and_prefix

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name


def bucket_name() -> str:
    return repo_name(prefix="buckets")


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch):
    """Set HF_TOKEN and HF_ENDPOINT for all CLI tests in this module."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HF_ENDPOINT", ENDPOINT_STAGING)
    yield


def cli(command: str, input: Optional[str] = None) -> Result:
    """
    Invoke a CLI command.

    Example:
        ```
        >>> cli("hf buckets create my-bucket")
        ```
    """
    assert command.startswith("hf ")
    args = command.split(" ")[1:]
    return CliRunner().invoke(app, [*args], input=input)


@pytest.fixture(scope="module")
def api() -> HfApi:
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture(scope="module")
def bucket_read(api: HfApi) -> str:
    """Module-scoped bucket for read-only tests (info, list)."""
    bucket_url = api.create_bucket(bucket_name())
    return bucket_url.bucket_id


@pytest.fixture
def bucket_write(api: HfApi) -> str:
    """Function-scoped bucket for destructive tests (delete)."""
    bucket_url = api.create_bucket(bucket_name())
    return bucket_url.bucket_id


# =============================================================================
# Create
# =============================================================================


def _handle_to_bucket_id(handle: str) -> str:
    """Extract bucket_id from a handle like 'hf://buckets/user/name'."""
    prefix = "hf://buckets/"
    if handle.startswith(prefix):
        return handle[len(prefix) :]
    return handle


@pytest.mark.parametrize(
    "path, expected",
    [
        ("namespace/bucket", ("namespace/bucket", "")),
        ("namespace/bucket/prefix", ("namespace/bucket", "prefix")),
        ("namespace/bucket/deep/nested/prefix", ("namespace/bucket", "deep/nested/prefix")),
        ("org/my-bucket/", ("org/my-bucket", "")),
    ],
)
def test_split_bucket_id_and_prefix(path: str, expected: tuple):
    assert _split_bucket_id_and_prefix(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "just-a-name",
        "",
        "/bucket",
        "namespace/",
    ],
)
def test_split_bucket_id_and_prefix_invalid(path: str):
    with pytest.raises(ValueError, match="Invalid bucket path"):
        _split_bucket_id_and_prefix(path)


def test_create_bucket(api: HfApi):
    name = bucket_name()
    result = cli(f"hf buckets create {name} --quiet")
    assert result.exit_code == 0
    handle = result.output.strip()
    assert handle == f"hf://buckets/{USER}/{name}"

    # Verify bucket exists
    bucket_id = _handle_to_bucket_id(handle)
    info = api.bucket_info(bucket_id)
    assert info.id == bucket_id


def test_create_bucket_private(api: HfApi):
    name = bucket_name()
    result = cli(f"hf buckets create {name} --private --quiet")
    assert result.exit_code == 0
    bucket_id = _handle_to_bucket_id(result.output.strip())

    info = api.bucket_info(bucket_id)
    assert info.private is True


def test_create_bucket_exist_ok():
    name = bucket_name()

    # First create succeeds
    result1 = cli(f"hf buckets create {name} --quiet")
    assert result1.exit_code == 0, result1.output

    # Second create without --exist-ok fails
    result2 = cli(f"hf buckets create {name} --quiet")
    assert result2.exit_code != 0
    assert isinstance(result2.exception, HfHubHTTPError)
    assert result2.exception.response.status_code == 409

    # Second create with --exist-ok succeeds
    result3 = cli(f"hf buckets create {name} --exist-ok --quiet")
    assert result3.exit_code == 0
    assert result3.output.strip() == f"hf://buckets/{USER}/{name}"


def test_create_bucket_with_hf_prefix(api: HfApi):
    name = bucket_name()
    hf_handle = f"hf://buckets/{USER}/{name}"
    result = cli(f"hf buckets create {hf_handle} --quiet")
    assert result.exit_code == 0

    assert result.output.strip() == hf_handle

    bucket_id = _handle_to_bucket_id(hf_handle)
    info = api.bucket_info(bucket_id)
    assert info.id == bucket_id


# =============================================================================
# Info
# =============================================================================


def test_bucket_info(bucket_read: str):
    result = cli(f"hf buckets info {bucket_read}")
    assert result.exit_code == 0

    data = json.loads(result.output)
    assert data["id"] == bucket_read
    assert data["private"] is False


def test_bucket_info_quiet(bucket_read: str):
    result = cli(f"hf buckets info {bucket_read} --quiet")
    assert result.exit_code == 0
    assert result.output.strip() == bucket_read


# =============================================================================
# Delete
# =============================================================================


def test_delete_bucket(api: HfApi, bucket_write: str):
    result = cli(f"hf buckets delete {bucket_write} --yes")
    assert result.exit_code == 0

    with pytest.raises(BucketNotFoundError):
        api.bucket_info(bucket_write)


def test_delete_bucket_missing_ok():
    nonexistent = f"{USER}/{bucket_name()}"
    result = cli(f"hf buckets delete {nonexistent} --yes --missing-ok")
    assert result.exit_code == 0


def test_delete_bucket_not_found():
    nonexistent = f"{USER}/{bucket_name()}"
    result = cli(f"hf buckets delete {nonexistent} --yes")
    assert result.exit_code != 0


# =============================================================================
# List buckets
# =============================================================================


def test_bucket_list_table(bucket_read: str):
    # Table format: just verify the command succeeds (table truncates IDs)
    result = cli("hf buckets list")
    assert result.exit_code == 0
    assert len(result.output.splitlines()) > 2  # return as table, ids are truncated


def test_bucket_list_json(bucket_read: str):
    result = cli("hf buckets list --format json")
    assert result.exit_code == 0

    buckets = json.loads(result.output)
    assert len(buckets) > 0
    assert bucket_read in {bucket["id"] for bucket in buckets}


def test_bucket_list_quiet(bucket_read: str):
    result = cli("hf buckets list --quiet")
    assert result.exit_code == 0

    ids = result.output.strip().splitlines()  # 1 id per line
    assert bucket_read in ids


def test_bucket_list_namespace(bucket_read: str):
    result = cli(f"hf buckets list {USER} --quiet")
    assert result.exit_code == 0

    ids = result.output.strip().splitlines()
    assert bucket_read in ids


def test_bucket_ls_alias(bucket_read: str):
    """'hf buckets ls' is an alias for 'hf buckets list'."""
    result = cli("hf buckets ls --quiet")
    assert result.exit_code == 0

    ids = result.output.strip().splitlines()
    assert bucket_read in ids


def test_bucket_list_namespace_with_hf_prefix(bucket_read: str):
    """hf://buckets/namespace format is treated as listing buckets."""
    result = cli(f"hf buckets list hf://buckets/{USER} --quiet")
    assert result.exit_code == 0

    ids = result.output.strip().splitlines()
    assert bucket_read in ids


def test_bucket_list_error_tree_with_namespace():
    """Cannot use --tree when listing buckets."""
    result = cli(f"hf buckets list {USER} --tree")
    assert result.exit_code != 0
    assert "Cannot use --tree when listing buckets" in result.output


def test_bucket_list_error_recursive_with_namespace():
    """Cannot use --recursive when listing buckets."""
    result = cli(f"hf buckets list {USER} --recursive")
    assert result.exit_code != 0
    assert "Cannot use --recursive when listing buckets" in result.output


# =============================================================================
# List files
# =============================================================================

# Fixed dates for exact output matching
MTIME_FIX = "2025-01-15 10:30:00"  # 19 chars, matches _format_mtime default format
MTIME_FIX_SHORT = "Jan 15 10:30"  # 12 chars, matches _format_mtime human_readable format

# Spaces for exact output matching
MTIME_GAP = " " * len(MTIME_FIX)
MTIME_GAP_SHORT = " " * len(MTIME_FIX_SHORT)


def _check_list_output(command: str, expected_lines: list[str]) -> None:
    """Run a `hf buckets list` command and assert output matches expected lines exactly."""
    result = cli(command)
    assert result.exit_code == 0
    actual = [line for line in result.output.splitlines() if line.strip()]
    assert actual == expected_lines


@pytest.fixture(scope="module")
def _populated_bucket(api: HfApi) -> str:
    """Module-scoped bucket with files for list files tests (root files, nested dirs)."""
    bucket_url = api.create_bucket(bucket_name())
    api.batch_bucket_files(
        bucket_url.bucket_id,
        add=[
            (b"hello", "file.txt"),
            (b"x" * 2048, "big.bin"),
            (b"nested content", "sub/nested.txt"),
            (b"deep", "sub/deep/file.txt"),
        ],
    )
    return bucket_url.bucket_id


@pytest.fixture
def tree_bucket(monkeypatch: pytest.MonkeyPatch, _populated_bucket: str) -> str:
    """Use populated bucket + make _format_mtime return a fixed date for exact output matching."""

    def _fixed(mtime, human_readable=False):
        if mtime is None:
            return ""
        return MTIME_FIX_SHORT if human_readable else MTIME_FIX

    monkeypatch.setattr("huggingface_hub.cli.buckets._format_mtime", _fixed)

    return _populated_bucket


def test_list_files_default(tree_bucket: str):
    """Default (non-recursive) table output shows root files and directories."""
    _check_list_output(
        f"hf buckets list {tree_bucket}",
        [
            f"        2048  {MTIME_FIX}  big.bin",
            f"           5  {MTIME_FIX}  file.txt",
            f"              {MTIME_FIX}  sub/",
        ],
    )


def test_list_files_recursive(tree_bucket: str):
    """Recursive listing shows all files including nested ones."""
    _check_list_output(
        f"hf buckets list {tree_bucket} -R",
        [
            f"        2048  {MTIME_FIX}  big.bin",
            f"           5  {MTIME_FIX}  file.txt",
            f"           4  {MTIME_FIX}  sub/deep/file.txt",
            f"          14  {MTIME_FIX}  sub/nested.txt",
        ],
    )


def test_list_files_human_readable(tree_bucket: str):
    """Human-readable sizes and short dates are shown with -h flag."""
    _check_list_output(
        f"hf buckets list {tree_bucket} -h -R",
        [
            f"      2.0 KB         {MTIME_FIX_SHORT}  big.bin",
            f"         5 B         {MTIME_FIX_SHORT}  file.txt",
            f"         4 B         {MTIME_FIX_SHORT}  sub/deep/file.txt",
            f"        14 B         {MTIME_FIX_SHORT}  sub/nested.txt",
        ],
    )


def test_list_files_tree_view(tree_bucket: str):
    """--tree -R renders ASCII tree with size+date before connectors."""
    _check_list_output(
        f"hf buckets list {tree_bucket} --tree -R",
        [
            f"2048  {MTIME_FIX}  ├── big.bin",
            f"   5  {MTIME_FIX}  ├── file.txt",
            f"      {MTIME_GAP}  └── sub/",
            f"      {MTIME_GAP}      ├── deep/",
            f"   4  {MTIME_FIX}      │   └── file.txt",
            f"  14  {MTIME_FIX}      └── nested.txt",
        ],
    )


def test_list_files_tree_view_non_recursive(tree_bucket: str):
    """--tree without -R only shows top-level entries."""
    _check_list_output(
        f"hf buckets list {tree_bucket} --tree",
        [
            f"2048  {MTIME_FIX}  ├── big.bin",
            f"   5  {MTIME_FIX}  ├── file.txt",
            f"      {MTIME_GAP}  └── sub/",
        ],
    )


def test_list_files_with_prefix(tree_bucket: str):
    """Passing a prefix only lists files under that prefix."""
    _check_list_output(
        f"hf buckets list {tree_bucket}/sub -R",
        [
            f"           4  {MTIME_FIX}  sub/deep/file.txt",
            f"          14  {MTIME_FIX}  sub/nested.txt",
        ],
    )


def test_list_files_with_hf_prefix(tree_bucket: str):
    """hf://buckets/ format works the same as short format."""
    _check_list_output(
        f"hf buckets list hf://buckets/{tree_bucket} -R",
        [
            f"        2048  {MTIME_FIX}  big.bin",
            f"           5  {MTIME_FIX}  file.txt",
            f"           4  {MTIME_FIX}  sub/deep/file.txt",
            f"          14  {MTIME_FIX}  sub/nested.txt",
        ],
    )


def test_list_files_with_hf_prefix_and_subprefix(tree_bucket: str):
    """hf://buckets/ format with a sub-prefix filters to that prefix."""
    _check_list_output(
        f"hf buckets list hf://buckets/{tree_bucket}/sub -R",
        [
            f"           4  {MTIME_FIX}  sub/deep/file.txt",
            f"          14  {MTIME_FIX}  sub/nested.txt",
        ],
    )


def test_list_files_empty_bucket(api: HfApi):
    """Empty bucket prints '(empty)'."""
    bucket_url = api.create_bucket(bucket_name())
    _check_list_output(f"hf buckets list {bucket_url.bucket_id}", ["(empty)"])


def test_list_files_quiet(tree_bucket: str):
    """--quiet prints one filename per line."""
    _check_list_output(
        f"hf buckets list {tree_bucket} -R --quiet",
        [
            "big.bin",
            "file.txt",
            "sub/deep/file.txt",
            "sub/nested.txt",
        ],
    )


def test_list_files_json(tree_bucket: str):
    """--format json outputs JSON array of file objects."""
    result = cli(f"hf buckets list {tree_bucket} -R --format json")
    assert result.exit_code == 0

    data = json.loads(result.output)
    assert isinstance(data, list)
    paths = {item["path"] for item in data}
    assert "big.bin" in paths
    assert "file.txt" in paths


def test_list_files_tree_with_human_readable(tree_bucket: str):
    """--tree -h shows human-readable sizes and short dates in tree format."""
    _check_list_output(
        f"hf buckets list {tree_bucket} --tree -R -h",
        [
            f"2.0 KB  {MTIME_FIX_SHORT}  ├── big.bin",
            f"   5 B  {MTIME_FIX_SHORT}  ├── file.txt",
            f"        {MTIME_GAP_SHORT}  └── sub/",
            f"        {MTIME_GAP_SHORT}      ├── deep/",
            f"   4 B  {MTIME_FIX_SHORT}      │   └── file.txt",
            f"  14 B  {MTIME_FIX_SHORT}      └── nested.txt",
        ],
    )


def test_list_files_tree_quiet(tree_bucket: str):
    """--tree --quiet shows only the tree structure without sizes/dates."""
    _check_list_output(
        f"hf buckets list {tree_bucket} --tree -R --quiet",
        [
            "├── big.bin",
            "├── file.txt",
            "└── sub/",
            "    ├── deep/",
            "    │   └── file.txt",
            "    └── nested.txt",
        ],
    )


def test_list_files_error_tree_with_json(tree_bucket: str):
    """Cannot use --tree with --format json."""
    result = cli(f"hf buckets list {tree_bucket} --tree --format json")
    assert result.exit_code != 0
    assert "Cannot use --tree with --format json" in result.output


# =============================================================================
# Cp
# =============================================================================


@pytest.fixture(scope="module")
def bucket_with_files(api: HfApi) -> str:
    """Module-scoped bucket pre-populated with files for cp download tests."""
    bucket_url = api.create_bucket(bucket_name())
    api.batch_bucket_files(
        bucket_url.bucket_id,
        add=[
            (b"hello", "file.txt"),
            (b"nested content", "sub/nested.txt"),
        ],
    )
    return bucket_url.bucket_id


# -- Upload tests --


def test_cp_upload_file_to_bucket_root(api: HfApi, tmp_path: Path):
    """Upload a local file to a bucket (no prefix)."""
    bucket_url = api.create_bucket(bucket_name())
    bucket_id = bucket_url.bucket_id

    local_file = tmp_path / "local.txt"
    local_file.write_text("upload me")

    result = cli(f"hf buckets cp {local_file} hf://buckets/{bucket_id}")
    assert result.exit_code == 0
    assert "Uploaded:" in result.output

    # Verify file exists in bucket with basename as remote path
    files = {f.path for f in api.list_bucket_tree(bucket_id)}
    assert "local.txt" in files


def test_cp_upload_file_to_bucket_prefix(api: HfApi, tmp_path: Path):
    """Upload a local file to a bucket subdirectory (trailing slash prefix)."""
    bucket_url = api.create_bucket(bucket_name())
    bucket_id = bucket_url.bucket_id

    local_file = tmp_path / "data.csv"
    local_file.write_text("a,b,c")

    result = cli(f"hf buckets cp {local_file} hf://buckets/{bucket_id}/logs/")
    assert result.exit_code == 0

    files = {f.path for f in api.list_bucket_tree(bucket_id)}
    assert "logs/data.csv" in files


def test_cp_upload_file_with_remote_name(api: HfApi, tmp_path: Path):
    """Upload a local file with an explicit remote filename."""
    bucket_url = api.create_bucket(bucket_name())
    bucket_id = bucket_url.bucket_id

    local_file = tmp_path / "original.txt"
    local_file.write_text("renamed upload")

    result = cli(f"hf buckets cp {local_file} hf://buckets/{bucket_id}/remote-name.txt")
    assert result.exit_code == 0

    files = {f.path for f in api.list_bucket_tree(bucket_id)}
    assert "remote-name.txt" in files


def test_cp_upload_file_quiet(api: HfApi, tmp_path: Path):
    """Upload with --quiet suppresses output."""
    bucket_url = api.create_bucket(bucket_name())
    bucket_id = bucket_url.bucket_id

    local_file = tmp_path / "quiet.txt"
    local_file.write_text("quiet")

    result = cli(f"hf buckets cp {local_file} hf://buckets/{bucket_id}/quiet.txt --quiet")
    assert result.exit_code == 0
    assert "Uploaded:" not in result.output


def test_cp_upload_from_stdin(api: HfApi):
    """Upload file content from stdin."""
    bucket_url = api.create_bucket(bucket_name())
    bucket_id = bucket_url.bucket_id

    result = cli(f"hf buckets cp - hf://buckets/{bucket_id}/from-stdin.txt", input="stdin data")
    assert result.exit_code == 0
    assert "Uploaded:" in result.output


# -- Download tests --


def test_cp_download_to_explicit_file(bucket_with_files: str, tmp_path: Path):
    """Download a bucket file to a specific local path."""
    output_file = tmp_path / "output.txt"
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt {output_file}")
    assert result.exit_code == 0
    assert "Downloaded:" in result.output
    assert output_file.read_text() == "hello"


def test_cp_download_to_directory(bucket_with_files: str, tmp_path: Path):
    """Download a bucket file to an existing directory (uses original filename)."""
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt {tmp_path}/")
    assert result.exit_code == 0

    downloaded = tmp_path / "file.txt"
    assert downloaded.exists()
    assert downloaded.read_text() == "hello"


def test_cp_download_nested_file(bucket_with_files: str, tmp_path: Path):
    """Download a file from a subdirectory in the bucket."""
    output_file = tmp_path / "nested.txt"
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/sub/nested.txt {output_file}")
    assert result.exit_code == 0
    assert output_file.read_text() == "nested content"


def test_cp_download_to_stdout(bucket_with_files: str):
    """Download a bucket file to stdout."""
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt -")
    assert result.exit_code == 0
    assert "hello" in result.output
    # stdout mode should NOT print "Downloaded:" message
    assert "Downloaded:" not in result.output


def test_cp_download_quiet(bucket_with_files: str, tmp_path: Path):
    """Download with --quiet suppresses the status message."""
    output_file = tmp_path / "quiet-download.txt"
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt {output_file} --quiet")
    assert result.exit_code == 0
    assert "Downloaded:" not in result.output
    assert output_file.read_text() == "hello"


def test_cp_download_creates_parent_dirs(bucket_with_files: str, tmp_path: Path):
    """Download creates parent directories when they don't exist."""
    output_file = tmp_path / "a" / "b" / "c" / "output.txt"
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt {output_file}")
    assert result.exit_code == 0
    assert output_file.read_text() == "hello"


# -- Validation error tests --


def test_cp_error_remote_to_remote():
    """Both src and dst are bucket paths."""
    result = cli("hf buckets cp hf://buckets/user/a/file.txt hf://buckets/user/b/file.txt")
    assert result.exit_code != 0


def test_cp_error_both_local(tmp_path: Path):
    """Both src and dst are local paths."""
    src = tmp_path / "src.txt"
    src.write_text("x")
    dst = tmp_path / "dst.txt"
    result = cli(f"hf buckets cp {src} {dst}")
    assert result.exit_code != 0
    assert "one of src or dst must be a bucket path" in result.output.lower()


def test_cp_error_missing_destination(tmp_path: Path):
    """Local src without a destination."""
    src = tmp_path / "orphan.txt"
    src.write_text("x")
    result = cli(f"hf buckets cp {src}")
    assert result.exit_code != 0
    assert "Missing destination" in result.output


def test_cp_error_stdin_without_bucket_dest():
    """Stdin upload requires a bucket destination."""
    result = cli("hf buckets cp - /tmp/local.txt", input="data")
    assert result.exit_code != 0
    assert "Stdin upload requires a bucket destination" in result.output


def test_cp_error_stdin_no_filename_empty_prefix():
    """Stdin upload to a bucket path without a filename (empty prefix)."""
    result = cli(f"hf buckets cp - hf://buckets/{USER}/some-bucket", input="data")
    assert result.exit_code != 0
    assert "full destination path including filename" in result.output


def test_cp_error_stdin_no_filename_trailing_slash():
    """Stdin upload to a bucket path with trailing slash (no filename)."""
    result = cli(f"hf buckets cp - hf://buckets/{USER}/some-bucket/logs/", input="data")
    assert result.exit_code != 0
    assert "full destination path including filename" in result.output


def test_cp_error_stdout_with_local_source(tmp_path: Path):
    """Cannot pipe to stdout when source is not a bucket path."""
    src = tmp_path / "local.txt"
    src.write_text("x")
    result = cli(f"hf buckets cp {src} -")
    assert result.exit_code != 0
    assert "one of src or dst must be a bucket path" in result.output.lower()


def test_cp_error_source_is_directory(tmp_path: Path):
    """Source must be a file, not a directory."""
    result = cli(f"hf buckets cp {tmp_path} hf://buckets/{USER}/some-bucket/file.txt")
    assert result.exit_code != 0
    assert "source must be a file, not a directory." in result.output.lower()


def test_cp_error_source_file_not_found():
    """Source file does not exist."""
    result = cli(f"hf buckets cp /nonexistent/file.txt hf://buckets/{USER}/some-bucket/file.txt")
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


# =============================================================================
# Sync
# =============================================================================


def _make_local_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a local directory with files.

    Args:
        tmp_path: pytest tmp_path fixture
        files: dict mapping relative path to file content

    Returns:
        Path to the created directory.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for rel_path, content in files.items():
        file_path = data_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    return data_dir


def _remote_files(api: HfApi, bucket_id: str) -> set[str]:
    """Return set of file paths in a bucket."""
    return {f.path for f in api.list_bucket_tree(bucket_id)}


# -- Upload tests --


def test_sync_upload_new_files(api: HfApi, bucket_write: str, tmp_path: Path):
    """Upload a local directory to an empty bucket."""
    data_dir = _make_local_dir(tmp_path, {"a.txt": "aaa", "b.txt": "bbb"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --quiet")
    assert result.exit_code == 0

    assert _remote_files(api, bucket_write) == {"a.txt", "b.txt"}


def test_sync_upload_skips_identical(bucket_write: str, tmp_path: Path):
    """Re-syncing unchanged files with --ignore-times prints 'Nothing to sync.' (sizes match)."""
    data_dir = _make_local_dir(tmp_path, {"a.txt": "aaa"})

    # First sync
    result1 = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --quiet")
    assert result1.exit_code == 0

    # Second sync with --ignore-times (sizes match → skip)
    result2 = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --ignore-times")
    assert result2.exit_code == 0
    assert "Nothing to sync." in result2.output


def test_sync_upload_skips_by_mtime(bucket_write: str, tmp_path: Path):
    """Re-syncing unchanged files uses mtime by default and skips already-uploaded files."""
    data_dir = _make_local_dir(tmp_path, {"a.txt": "aaa", "b.txt": "bbb"})

    # First sync uploads both files
    result1 = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --verbose --quiet")
    assert result1.exit_code == 0
    assert "Uploading: a.txt" in result1.output
    assert "Uploading: b.txt" in result1.output

    # Second sync (default mtime comparison) should skip everything
    result2 = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write}")
    assert result2.exit_code == 0
    assert "Nothing to sync." in result2.output


def test_sync_upload_with_prefix(api: HfApi, bucket_write: str, tmp_path: Path):
    """Upload files under a bucket prefix."""
    data_dir = _make_local_dir(tmp_path, {"f.txt": "data"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write}/subdir --quiet")
    assert result.exit_code == 0

    assert "subdir/f.txt" in _remote_files(api, bucket_write)


def test_sync_upload_prefix_with_trailing_slash(api: HfApi, bucket_write: str, tmp_path: Path):
    """Trailing slash in prefix should not cause double slashes in remote paths.

    This is a regression test for a bug where paths like 'hf://buckets/user/bucket/sub/'
    would result in remote paths like 'sub//file.txt' instead of 'sub/file.txt'.
    """
    data_dir = _make_local_dir(tmp_path, {"trailing.txt": "data"})

    # Use trailing slash in destination path
    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write}/sub/ --quiet")
    assert result.exit_code == 0

    # Verify file is at correct path (no double slashes)
    remote = _remote_files(api, bucket_write)
    assert "sub/trailing.txt" in remote
    assert "sub//trailing.txt" not in remote


def test_sync_download_prefix_does_not_match_similar_names(api: HfApi, bucket_write: str, tmp_path: Path):
    """Sync with prefix should not match files that share prefix string without directory boundary.

    This is a regression test for a bug where prefix stripping used path.startswith(prefix)
    which would incorrectly match "submarine.txt" when prefix="sub", yielding "marine.txt"
    as the relative path instead of keeping the full path.
    """
    # Upload files: one under sub/ directory, one with similar prefix at root
    api.batch_bucket_files(
        bucket_write,
        add=[
            (b"in subdir", "sub/file.txt"),
            (b"similar name", "submarine.txt"),
        ],
    )

    download_dir = tmp_path / "download"
    download_dir.mkdir()

    # Sync only the sub/ prefix
    result = cli(f"hf buckets sync hf://buckets/{bucket_write}/sub {download_dir} --quiet")
    assert result.exit_code == 0

    # Should download file.txt (from sub/)
    assert (download_dir / "file.txt").exists()
    assert (download_dir / "file.txt").read_text() == "in subdir"

    # Should NOT download submarine.txt or incorrectly-named marine.txt
    assert not (download_dir / "marine.txt").exists()
    assert not (download_dir / "submarine.txt").exists()


def test_sync_upload_verbose(bucket_write: str, tmp_path: Path):
    """--verbose shows per-file operation lines."""
    data_dir = _make_local_dir(tmp_path, {"v.txt": "verbose"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --verbose --quiet")
    assert result.exit_code == 0
    assert "Uploading: v.txt" in result.output


def test_sync_upload_quiet(bucket_write: str, tmp_path: Path):
    """--quiet produces no output."""
    data_dir = _make_local_dir(tmp_path, {"q.txt": "quiet"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --quiet")
    assert result.exit_code == 0
    assert result.output.strip() == ""


# -- Download tests --


def test_sync_download_new_files(api: HfApi, bucket_write: str, tmp_path: Path):
    """Download bucket files to a local directory."""
    api.batch_bucket_files(bucket_write, add=[(b"hello", "file.txt"), (b"world", "other.txt")])

    dest = tmp_path / "download"
    result = cli(f"hf buckets sync hf://buckets/{bucket_write} {dest} --quiet")
    assert result.exit_code == 0

    assert (dest / "file.txt").read_text() == "hello"
    assert (dest / "other.txt").read_text() == "world"


def test_sync_download_skips_identical(api: HfApi, bucket_write: str, tmp_path: Path):
    """Re-syncing a download with no changes prints 'Nothing to sync.'."""
    api.batch_bucket_files(bucket_write, add=[(b"data", "f.txt")])

    dest = tmp_path / "download"
    # First sync
    result1 = cli(f"hf buckets sync hf://buckets/{bucket_write} {dest} --quiet")
    assert result1.exit_code == 0

    # Second sync
    result2 = cli(f"hf buckets sync hf://buckets/{bucket_write} {dest}")
    assert result2.exit_code == 0
    assert "Nothing to sync." in result2.output


def test_sync_download_creates_subdirs(api: HfApi, bucket_write: str, tmp_path: Path):
    """Download creates local subdirectories for nested files."""
    api.batch_bucket_files(bucket_write, add=[(b"nested", "a/b/c.txt")])

    dest = tmp_path / "download"
    result = cli(f"hf buckets sync hf://buckets/{bucket_write} {dest} --quiet")
    assert result.exit_code == 0

    assert (dest / "a" / "b" / "c.txt").read_text() == "nested"


# -- --delete option --


def test_sync_upload_delete(api: HfApi, bucket_write: str, tmp_path: Path):
    """Upload with --delete removes remote files not in local."""
    # Pre-populate remote with an extra file
    api.batch_bucket_files(bucket_write, add=[(b"old", "old.txt"), (b"keep", "keep.txt")])

    # Local only has keep.txt
    data_dir = _make_local_dir(tmp_path, {"keep.txt": "keep"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --delete --quiet")
    assert result.exit_code == 0

    files = _remote_files(api, bucket_write)
    assert "keep.txt" in files
    assert "old.txt" not in files


def test_sync_download_delete(api: HfApi, bucket_write: str, tmp_path: Path):
    """Download with --delete removes local files not in remote."""
    api.batch_bucket_files(bucket_write, add=[(b"keep", "keep.txt")])

    # Local has an extra file
    dest = tmp_path / "download"
    dest.mkdir()
    (dest / "extra.txt").write_text("should be deleted")
    (dest / "keep.txt").write_text("keep")

    result = cli(f"hf buckets sync hf://buckets/{bucket_write} {dest} --delete --quiet")
    assert result.exit_code == 0

    assert (dest / "keep.txt").exists()
    assert not (dest / "extra.txt").exists()


# -- Comparison mode options --


def test_sync_upload_ignore_existing(api: HfApi, bucket_write: str, tmp_path: Path):
    """--ignore-existing skips files already in remote."""
    # Pre-populate remote with existing.txt (old content)
    api.batch_bucket_files(bucket_write, add=[(b"old", "existing.txt")])

    # Local has both existing.txt (new content) and new.txt
    data_dir = _make_local_dir(tmp_path, {"existing.txt": "new", "new.txt": "brand new"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --ignore-existing --quiet")
    assert result.exit_code == 0

    # new.txt should be uploaded, existing.txt should be skipped (old content preserved)
    files = _remote_files(api, bucket_write)
    assert "new.txt" in files
    assert "existing.txt" in files


def test_sync_upload_existing_only(api: HfApi, bucket_write: str, tmp_path: Path):
    """--existing skips new files (only updates existing)."""
    # Pre-populate remote with only existing.txt
    api.batch_bucket_files(bucket_write, add=[(b"old", "existing.txt")])

    # Local has both existing.txt and new.txt
    data_dir = _make_local_dir(tmp_path, {"existing.txt": "updated", "new.txt": "should not appear"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --existing --quiet")
    assert result.exit_code == 0

    # new.txt should NOT be uploaded
    files = _remote_files(api, bucket_write)
    assert "existing.txt" in files
    assert "new.txt" not in files


# -- --plan and --apply workflow --


def test_sync_plan_saves_file(bucket_write: str, tmp_path: Path):
    """--plan saves a JSONL plan file and prints summary."""
    data_dir = _make_local_dir(tmp_path, {"p.txt": "plan me"})
    plan_file = tmp_path / "plan.jsonl"

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --plan {plan_file}")
    assert result.exit_code == 0
    assert "Plan saved to:" in result.output

    # Verify plan file structure
    lines = plan_file.read_text().strip().splitlines()
    assert len(lines) == 2  # header + 1 operation

    header = json.loads(lines[0])
    assert header["type"] == "header"
    assert header["summary"]["uploads"] == 1

    op = json.loads(lines[1])
    assert op["type"] == "operation"
    assert op["action"] == "upload"
    assert op["path"] == "p.txt"


def test_sync_plan_then_apply(api: HfApi, bucket_write: str, tmp_path: Path):
    """End-to-end: plan → review → apply → verify."""
    data_dir = _make_local_dir(tmp_path, {"x.txt": "x", "y.txt": "y"})
    plan_file = tmp_path / "plan.jsonl"

    # Plan (nothing executed)
    cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --plan {plan_file} --quiet")
    assert _remote_files(api, bucket_write) == set()  # nothing uploaded yet

    # Apply
    result = cli(f"hf buckets sync --apply {plan_file} --quiet")
    assert result.exit_code == 0

    assert _remote_files(api, bucket_write) == {"x.txt", "y.txt"}


# -- --dry-run --


def test_sync_dry_run(api: HfApi, bucket_write: str, tmp_path: Path):
    """--dry-run prints JSONL plan to stdout without executing."""
    data_dir = _make_local_dir(tmp_path, {"a.txt": "aaa", "b.txt": "bbb"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --dry-run")
    assert result.exit_code == 0

    lines = result.output.strip().splitlines()
    assert len(lines) == 3  # header + 2 operations

    header = json.loads(lines[0])
    assert header["type"] == "header"
    assert header["summary"]["uploads"] == 2

    ops = [json.loads(line) for line in lines[1:]]
    assert all(op["type"] == "operation" for op in ops)
    assert all(op["action"] == "upload" for op in ops)
    paths = {op["path"] for op in ops}
    assert paths == {"a.txt", "b.txt"}

    # Verify nothing was actually uploaded
    assert len(_remote_files(api, bucket_write)) == 0


# -- Filtering --


def test_sync_upload_include(api: HfApi, bucket_write: str, tmp_path: Path):
    """--include only syncs matching files."""
    data_dir = _make_local_dir(tmp_path, {"keep.txt": "keep", "skip.csv": "skip"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --include *.txt --quiet")
    assert result.exit_code == 0

    files = _remote_files(api, bucket_write)
    assert "keep.txt" in files
    assert "skip.csv" not in files


def test_sync_upload_exclude(api: HfApi, bucket_write: str, tmp_path: Path):
    """--exclude skips matching files."""
    data_dir = _make_local_dir(tmp_path, {"keep.txt": "keep", "skip.log": "skip"})

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --exclude *.log --quiet")
    assert result.exit_code == 0

    files = _remote_files(api, bucket_write)
    assert "keep.txt" in files
    assert "skip.log" not in files


def test_sync_upload_filter_from(api: HfApi, bucket_write: str, tmp_path: Path):
    """--filter-from reads include/exclude rules from a file."""
    data_dir = _make_local_dir(tmp_path, {"keep.txt": "keep", "skip.log": "skip", "also.csv": "also"})

    filter_file = tmp_path / "filters.txt"
    filter_file.write_text("# comment\n- *.log\n+ *.txt\n")

    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --filter-from {filter_file} --quiet")
    assert result.exit_code == 0

    files = _remote_files(api, bucket_write)
    assert "keep.txt" in files
    assert "skip.log" not in files
    # also.csv doesn't match any rule → included by default
    assert "also.csv" in files


# -- Validation errors --


def test_sync_top_level_alias(api: HfApi, bucket_write: str, tmp_path: Path):
    """'hf sync' is an alias for 'hf buckets sync'."""
    data_dir = _make_local_dir(tmp_path, {"alias.txt": "via top-level"})

    result = cli(f"hf sync {data_dir} hf://buckets/{bucket_write} --quiet")
    assert result.exit_code == 0

    assert "alias.txt" in _remote_files(api, bucket_write)


def test_sync_error_no_args():
    """No source/dest raises an error."""
    result = cli("hf buckets sync")
    assert result.exit_code != 0


def test_sync_error_remote_to_remote():
    """Both paths are bucket paths."""
    result = cli("hf buckets sync hf://buckets/u/a hf://buckets/u/b")
    assert result.exit_code != 0


def test_sync_error_both_local(tmp_path: Path):
    """Both paths are local."""
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    result = cli(f"hf buckets sync {src} {dst}")
    assert result.exit_code != 0


def test_sync_error_source_not_directory(tmp_path: Path):
    """Local source must be a directory."""
    src = tmp_path / "file.txt"
    src.write_text("x")
    result = cli(f"hf buckets sync {src} hf://buckets/{USER}/some-bucket")
    assert result.exit_code != 0


def test_sync_error_ignore_times_and_sizes(tmp_path: Path):
    """Cannot specify both --ignore-times and --ignore-sizes."""
    src = tmp_path / "src"
    src.mkdir()
    result = cli(f"hf buckets sync {src} hf://buckets/{USER}/b --ignore-times --ignore-sizes")
    assert result.exit_code != 0


def test_sync_error_existing_and_ignore_existing(tmp_path: Path):
    """Cannot specify both --existing and --ignore-existing."""
    src = tmp_path / "src"
    src.mkdir()
    result = cli(f"hf buckets sync {src} hf://buckets/{USER}/b --existing --ignore-existing")
    assert result.exit_code != 0


def test_sync_error_apply_with_source():
    """Cannot specify source when using --apply."""
    result = cli("hf buckets sync ./data --apply plan.jsonl")
    assert result.exit_code != 0


def test_sync_error_apply_with_plan():
    """Cannot specify both --apply and --plan."""
    result = cli("hf buckets sync --apply plan.jsonl --plan other.jsonl")
    assert result.exit_code != 0


def test_sync_error_apply_with_delete():
    """Cannot specify --delete when using --apply."""
    result = cli("hf buckets sync --apply plan.jsonl --delete")
    assert result.exit_code != 0


def test_sync_error_apply_with_include():
    """Cannot specify --include when using --apply."""
    result = cli("hf buckets sync --apply plan.jsonl --include *.txt")
    assert result.exit_code != 0


def test_sync_error_apply_with_exclude():
    """Cannot specify --exclude when using --apply."""
    result = cli("hf buckets sync --apply plan.jsonl --exclude *.log")
    assert result.exit_code != 0


def test_sync_error_dry_run_with_apply():
    """Cannot specify --dry-run when using --apply."""
    result = cli("hf buckets sync --apply plan.jsonl --dry-run")
    assert result.exit_code != 0


def test_sync_error_dry_run_with_plan(tmp_path: Path):
    """Cannot specify both --dry-run and --plan."""
    src = tmp_path / "src"
    src.mkdir()
    result = cli(f"hf buckets sync {src} hf://buckets/{USER}/b --dry-run --plan out.jsonl")
    assert result.exit_code != 0


def test_sync_upload_to_nonexistent_bucket_plans_all_files(tmp_path: Path):
    """Syncing to a non-existent bucket should treat remote as empty.

    This is a regression test for a bug where the sync code caught
    RepositoryNotFoundError instead of BucketNotFoundError, causing syncs
    to non-existent buckets to crash instead of gracefully treating the
    remote as empty.
    """
    data_dir = _make_local_dir(tmp_path, {"new.txt": "content", "other.txt": "data"})

    # Use --dry-run to avoid actually creating the bucket
    # The key is that the sync should not crash with BucketNotFoundError
    nonexistent_bucket = f"{USER}/{bucket_name()}"
    result = cli(f"hf buckets sync {data_dir} hf://buckets/{nonexistent_bucket} --dry-run")

    # Should succeed (exit code 0) and plan to upload all files
    assert result.exit_code == 0

    # Should have a header + 2 file operations in the output
    import json

    lines = result.output.strip().splitlines()
    assert len(lines) == 3  # header + 2 operations

    operations = [json.loads(line) for line in lines[1:]]
    paths = {op["path"] for op in operations}
    assert paths == {"new.txt", "other.txt"}
    assert all(op["action"] == "upload" for op in operations)
    assert all(op["reason"] == "new file" for op in operations)


def test_sync_ignore_sizes_skip_reason_shows_dest_newer(api: HfApi, bucket_write: str, tmp_path: Path):
    """Sync skip reason should accurately say 'remote newer' or 'local newer' instead of 'same mtime'.

    This is a regression test for a bug where the skip reason when using --ignore-sizes
    always said "same mtime" even when the destination was actually newer than the source.
    """
    import os
    import time

    # Upload a file to the bucket
    data_dir = _make_local_dir(tmp_path, {"file.txt": "content"})
    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --quiet")
    assert result.exit_code == 0

    # Wait a bit to ensure remote mtime is set
    time.sleep(1.5)

    # Set local file mtime to be significantly older than remote (more than 1s window)
    local_file = data_dir / "file.txt"
    old_mtime = time.time() - 10  # 10 seconds in the past
    os.utime(local_file, (old_mtime, old_mtime))

    # Sync with --ignore-sizes --dry-run (upload direction)
    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --ignore-sizes --dry-run")
    assert result.exit_code == 0

    # Parse the dry-run output
    lines = result.output.strip().splitlines()
    assert len(lines) >= 2  # header + at least 1 operation

    op = json.loads(lines[1])
    assert op["action"] == "skip"
    assert op["path"] == "file.txt"
    # The reason should be "remote newer", not "same mtime"
    assert op["reason"] == "remote newer", f"Expected 'remote newer' but got '{op['reason']}'"


def test_sync_ignore_sizes_download_skip_reason_shows_dest_newer(api: HfApi, bucket_write: str, tmp_path: Path):
    """Download sync skip reason should accurately say 'local newer' instead of 'same mtime'.

    This is a regression test for the download direction of the same bug where skip reason
    always said "same mtime" even when local file was actually newer than remote.
    """
    import os
    import time

    # Upload a file to the bucket
    data_dir = _make_local_dir(tmp_path, {"file.txt": "content"})
    result = cli(f"hf buckets sync {data_dir} hf://buckets/{bucket_write} --quiet")
    assert result.exit_code == 0

    # Wait for remote to be stable
    time.sleep(1.5)

    # Create download directory with a file that has a newer mtime
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    local_file = download_dir / "file.txt"
    local_file.write_text("content")

    # Set local file mtime to be significantly newer than remote (more than 1s window)
    new_mtime = time.time() + 10  # 10 seconds in the future
    os.utime(local_file, (new_mtime, new_mtime))

    # Sync with --ignore-sizes --dry-run (download direction)
    result = cli(f"hf buckets sync hf://buckets/{bucket_write} {download_dir} --ignore-sizes --dry-run")
    assert result.exit_code == 0

    # Parse the dry-run output
    lines = result.output.strip().splitlines()
    assert len(lines) >= 2  # header + at least 1 operation

    op = json.loads(lines[1])
    assert op["action"] == "skip"
    assert op["path"] == "file.txt"
    # The reason should be "local newer", not "same mtime"
    assert op["reason"] == "local newer", f"Expected 'local newer' but got '{op['reason']}'"
