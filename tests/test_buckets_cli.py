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
from typing import Optional

import pytest
from typer.testing import CliRunner, Result

from huggingface_hub import HfApi
from huggingface_hub.cli.hf import app
from huggingface_hub.errors import BucketNotFoundError, HfHubHTTPError

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
# List
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


def test_cp_upload_file_to_bucket_root(api: HfApi, tmp_path):
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


def test_cp_upload_file_to_bucket_prefix(api: HfApi, tmp_path):
    """Upload a local file to a bucket subdirectory (trailing slash prefix)."""
    bucket_url = api.create_bucket(bucket_name())
    bucket_id = bucket_url.bucket_id

    local_file = tmp_path / "data.csv"
    local_file.write_text("a,b,c")

    result = cli(f"hf buckets cp {local_file} hf://buckets/{bucket_id}/logs/")
    assert result.exit_code == 0

    files = {f.path for f in api.list_bucket_tree(bucket_id)}
    assert "logs/data.csv" in files


def test_cp_upload_file_with_remote_name(api: HfApi, tmp_path):
    """Upload a local file with an explicit remote filename."""
    bucket_url = api.create_bucket(bucket_name())
    bucket_id = bucket_url.bucket_id

    local_file = tmp_path / "original.txt"
    local_file.write_text("renamed upload")

    result = cli(f"hf buckets cp {local_file} hf://buckets/{bucket_id}/remote-name.txt")
    assert result.exit_code == 0

    files = {f.path for f in api.list_bucket_tree(bucket_id)}
    assert "remote-name.txt" in files


def test_cp_upload_file_quiet(api: HfApi, tmp_path):
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


def test_cp_download_to_explicit_file(bucket_with_files: str, tmp_path):
    """Download a bucket file to a specific local path."""
    output_file = tmp_path / "output.txt"
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt {output_file}")
    assert result.exit_code == 0
    assert "Downloaded:" in result.output
    assert output_file.read_text() == "hello"


def test_cp_download_to_directory(bucket_with_files: str, tmp_path):
    """Download a bucket file to an existing directory (uses original filename)."""
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt {tmp_path}/")
    assert result.exit_code == 0

    downloaded = tmp_path / "file.txt"
    assert downloaded.exists()
    assert downloaded.read_text() == "hello"


def test_cp_download_nested_file(bucket_with_files: str, tmp_path):
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


def test_cp_download_quiet(bucket_with_files: str, tmp_path):
    """Download with --quiet suppresses the status message."""
    output_file = tmp_path / "quiet-download.txt"
    result = cli(f"hf buckets cp hf://buckets/{bucket_with_files}/file.txt {output_file} --quiet")
    assert result.exit_code == 0
    assert "Downloaded:" not in result.output
    assert output_file.read_text() == "hello"


def test_cp_download_creates_parent_dirs(bucket_with_files: str, tmp_path):
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


def test_cp_error_both_local(tmp_path):
    """Both src and dst are local paths."""
    src = tmp_path / "src.txt"
    src.write_text("x")
    dst = tmp_path / "dst.txt"
    result = cli(f"hf buckets cp {src} {dst}")
    assert result.exit_code != 0
    assert "one of src or dst must be a bucket path" in result.output.lower()


def test_cp_error_missing_destination(tmp_path):
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


def test_cp_error_stdout_with_local_source(tmp_path):
    """Cannot pipe to stdout when source is not a bucket path."""
    src = tmp_path / "local.txt"
    src.write_text("x")
    result = cli(f"hf buckets cp {src} -")
    assert result.exit_code != 0
    assert "one of src or dst must be a bucket path" in result.output.lower()


def test_cp_error_source_is_directory(tmp_path):
    """Source must be a file, not a directory."""
    result = cli(f"hf buckets cp {tmp_path} hf://buckets/{USER}/some-bucket/file.txt")
    assert result.exit_code != 0
    assert "source must be a file, not a directory." in result.output.lower()


def test_cp_error_source_file_not_found():
    """Source file does not exist."""
    result = cli(f"hf buckets cp /nonexistent/file.txt hf://buckets/{USER}/some-bucket/file.txt")
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
