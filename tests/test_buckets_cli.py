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

import pytest
from typer.testing import CliRunner

from huggingface_hub import HfApi
from huggingface_hub.cli.hf import app
from huggingface_hub.errors import BucketNotFoundError, HfHubHTTPError

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name


def bucket_name() -> str:
    return repo_name(prefix="bucket")


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch):
    """Set HF_TOKEN and HF_ENDPOINT for all CLI tests in this module."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HF_ENDPOINT", ENDPOINT_STAGING)
    yield


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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


def test_create_bucket(runner: CliRunner, api: HfApi):
    name = bucket_name()
    result = runner.invoke(app, ["bucket", "create", name, "--quiet"])
    assert result.exit_code == 0
    handle = result.output.strip()
    assert handle == f"hf://buckets/{USER}/{name}"

    # Verify bucket exists
    bucket_id = _handle_to_bucket_id(handle)
    info = api.bucket_info(bucket_id)
    assert info.id == bucket_id


def test_create_bucket_private(runner: CliRunner, api: HfApi):
    name = bucket_name()
    result = runner.invoke(app, ["bucket", "create", name, "--private", "--quiet"])
    assert result.exit_code == 0, result.output
    bucket_id = _handle_to_bucket_id(result.output.strip())

    info = api.bucket_info(bucket_id)
    assert info.private is True


def test_create_bucket_exist_ok(runner: CliRunner):
    name = bucket_name()

    # First create succeeds
    result1 = runner.invoke(app, ["bucket", "create", name, "--quiet"])
    assert result1.exit_code == 0, result1.output

    # Second create without --exist-ok fails
    result2 = runner.invoke(app, ["bucket", "create", name, "--quiet"])
    assert result2.exit_code != 0
    assert isinstance(result2.exception, HfHubHTTPError)
    assert result2.exception.response.status_code == 409

    # Second create with --exist-ok succeeds
    result3 = runner.invoke(app, ["bucket", "create", name, "--exist-ok", "--quiet"])
    assert result3.exit_code == 0
    assert result3.output.strip() == f"hf://buckets/{USER}/{name}"


def test_create_bucket_with_hf_prefix(runner: CliRunner, api: HfApi):
    name = bucket_name()
    hf_handle = f"hf://buckets/{USER}/{name}"
    result = runner.invoke(app, ["bucket", "create", hf_handle, "--quiet"])
    assert result.exit_code == 0, result.output

    assert result.output.strip() == hf_handle

    bucket_id = _handle_to_bucket_id(hf_handle)
    info = api.bucket_info(bucket_id)
    assert info.id == bucket_id


# =============================================================================
# Info
# =============================================================================


def test_bucket_info(runner: CliRunner, bucket_read: str):
    result = runner.invoke(app, ["bucket", "info", bucket_read])
    assert result.exit_code == 0, result.output

    data = json.loads(result.output)
    assert data["id"] == bucket_read
    assert data["private"] is False


def test_bucket_info_quiet(runner: CliRunner, bucket_read: str):
    result = runner.invoke(app, ["bucket", "info", bucket_read, "--quiet"])
    assert result.exit_code == 0
    assert result.output.strip() == bucket_read


# =============================================================================
# List
# =============================================================================


def test_bucket_list_table(runner: CliRunner, bucket_read: str):
    # Table format: just verify the command succeeds (table truncates IDs)
    result = runner.invoke(app, ["bucket", "list"])
    assert result.exit_code == 0
    assert len(result.output.splitlines()) > 2  # return as table, ids are truncated


def test_bucket_list_json(runner: CliRunner, bucket_read: str):
    result = runner.invoke(app, ["bucket", "list", "--format", "json"])
    assert result.exit_code == 0

    buckets = json.loads(result.output)
    assert len(buckets) > 0
    assert bucket_read in {bucket["id"] for bucket in buckets}


def test_bucket_list_quiet(runner: CliRunner, bucket_read: str):
    result = runner.invoke(app, ["bucket", "list", "--quiet"])
    assert result.exit_code == 0

    ids = result.output.strip().splitlines()  # 1 id per line
    assert bucket_read in ids


def test_bucket_list_namespace(runner: CliRunner, bucket_read: str):
    result = runner.invoke(app, ["bucket", "list", USER, "--quiet"])
    assert result.exit_code == 0

    ids = result.output.strip().splitlines()
    assert bucket_read in ids


# =============================================================================
# Delete
# =============================================================================


def test_delete_bucket(runner: CliRunner, api: HfApi, bucket_write: str):
    result = runner.invoke(app, ["bucket", "delete", bucket_write, "--yes"])
    assert result.exit_code == 0, result.output

    with pytest.raises(BucketNotFoundError):
        api.bucket_info(bucket_write)


def test_delete_bucket_missing_ok(runner: CliRunner):
    nonexistent = f"{USER}/{bucket_name()}"
    result = runner.invoke(app, ["bucket", "delete", nonexistent, "--yes", "--missing-ok"])
    assert result.exit_code == 0


def test_delete_bucket_not_found(runner: CliRunner):
    nonexistent = f"{USER}/{bucket_name()}"
    result = runner.invoke(app, ["bucket", "delete", nonexistent, "--yes"])
    assert result.exit_code != 0
