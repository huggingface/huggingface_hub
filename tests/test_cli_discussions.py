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
import shlex
from typing import Optional

import pytest
from typer.testing import CliRunner, Result

from huggingface_hub import HfApi
from huggingface_hub.cli.hf import app

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch):
    """Set HF_TOKEN and HF_ENDPOINT for all CLI tests in this module."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HF_ENDPOINT", ENDPOINT_STAGING)
    yield


def cli(command: str, input: Optional[str] = None) -> Result:
    """Invoke a CLI command.

    Uses shlex.split to properly handle quoted arguments.

    Example:
        ```
        >>> cli('hf discussions create user/repo --title "My title"')
        ```
    """
    assert command.startswith("hf ")
    args = shlex.split(command)[1:]
    return CliRunner().invoke(app, [*args], input=input)


@pytest.fixture(scope="module")
def api() -> HfApi:
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture(scope="module")
def repo_with_discussion(api: HfApi) -> tuple:
    """Module-scoped repo with a discussion and a PR for read tests."""
    repo_id = api.create_repo(repo_name(prefix="discussions"), exist_ok=True).repo_id
    discussion = api.create_discussion(repo_id=repo_id, title="Test discussion")
    pr = api.create_pull_request(repo_id=repo_id, title="Test PR")
    return repo_id, discussion.num, pr.num


@pytest.fixture
def repo_for_write(api: HfApi) -> str:
    """Function-scoped repo for destructive tests."""
    return api.create_repo(repo_name(prefix="discussions"), exist_ok=True).repo_id


# =============================================================================
# List
# =============================================================================


def test_list_discussions(repo_with_discussion: tuple):
    repo_id, _, _ = repo_with_discussion
    result = cli(f"hf discussions list {repo_id}")
    assert result.exit_code == 0, result.output
    assert "#" in result.output


def test_list_discussions_quiet(repo_with_discussion: tuple):
    repo_id, disc_num, pr_num = repo_with_discussion
    result = cli(f"hf discussions list {repo_id} --status all --quiet")
    assert result.exit_code == 0, result.output
    nums = result.output.strip().splitlines()
    assert str(disc_num) in nums
    assert str(pr_num) in nums


def test_list_discussions_json(repo_with_discussion: tuple):
    repo_id, _, _ = repo_with_discussion
    result = cli(f"hf discussions list {repo_id} --status all --format json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) >= 2
    assert "num" in data[0]
    assert "title" in data[0]


def test_list_filter_kind_discussion(repo_with_discussion: tuple):
    repo_id, disc_num, pr_num = repo_with_discussion
    result = cli(f"hf discussions list {repo_id} --status all --kind discussion --quiet")
    assert result.exit_code == 0, result.output
    nums = result.output.strip().splitlines()
    assert str(disc_num) in nums
    assert str(pr_num) not in nums


def test_list_filter_kind_pull_request(repo_with_discussion: tuple):
    repo_id, disc_num, pr_num = repo_with_discussion
    result = cli(f"hf discussions list {repo_id} --status all --kind pull_request --quiet")
    assert result.exit_code == 0, result.output
    nums = result.output.strip().splitlines()
    assert str(pr_num) in nums
    assert str(disc_num) not in nums


def test_list_filter_status_closed(repo_with_discussion: tuple):
    repo_id, _, _ = repo_with_discussion
    result = cli(f"hf discussions list {repo_id} --status closed --quiet")
    assert result.exit_code == 0
    # All our test discussions are open, so closed should return nothing
    assert result.output.strip() == ""


# =============================================================================
# View
# =============================================================================


def test_view_discussion(repo_with_discussion: tuple):
    repo_id, disc_num, _ = repo_with_discussion
    result = cli(f"hf discussions view {repo_id} {disc_num}")
    assert result.exit_code == 0, result.output
    assert "Test discussion" in result.output
    assert f"#{disc_num}" in result.output
    assert "View on Hub:" in result.output


def test_view_pr(repo_with_discussion: tuple):
    repo_id, _, pr_num = repo_with_discussion
    result = cli(f"hf discussions view {repo_id} {pr_num}")
    assert result.exit_code == 0, result.output
    assert "Test PR" in result.output
    assert "Pull Request" in result.output


def test_view_json(repo_with_discussion: tuple):
    repo_id, disc_num, _ = repo_with_discussion
    result = cli(f"hf discussions view {repo_id} {disc_num} --format json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["num"] == disc_num
    assert data["title"] == "Test discussion"


def test_view_no_color(repo_with_discussion: tuple):
    repo_id, disc_num, _ = repo_with_discussion
    result = cli(f"hf discussions view {repo_id} {disc_num} --no-color")
    assert result.exit_code == 0, result.output
    assert "\u001b[" not in result.output


# =============================================================================
# Create
# =============================================================================


def test_create_discussion(repo_for_write: str):
    result = cli(f'hf discussions create {repo_for_write} --title "CLI test discussion"')
    assert result.exit_code == 0, result.output
    assert "Created discussion" in result.output
    assert "#" in result.output


def test_create_pull_request(repo_for_write: str):
    result = cli(f'hf discussions create {repo_for_write} --title "CLI test PR" --pull-request')
    assert result.exit_code == 0, result.output
    assert "Created pull request" in result.output
    assert "refs/pr/" in result.output


def test_create_with_body(repo_for_write: str):
    result = cli(f'hf discussions create {repo_for_write} --title "With body" --body "Some description"')
    assert result.exit_code == 0, result.output
    assert "Created discussion" in result.output


def test_create_with_body_file(repo_for_write: str, tmp_path):
    body_file = tmp_path / "body.md"
    body_file.write_text("Body from file")
    result = cli(f'hf discussions create {repo_for_write} --title "From file" --body-file {body_file}')
    assert result.exit_code == 0, result.output
    assert "Created discussion" in result.output


def test_create_body_and_body_file_conflict(repo_for_write: str, tmp_path):
    body_file = tmp_path / "body.md"
    body_file.write_text("Body from file")
    result = cli(f'hf discussions create {repo_for_write} --title "Conflict" --body "inline" --body-file {body_file}')
    assert result.exit_code != 0


# =============================================================================
# Comment
# =============================================================================


def test_comment_discussion(api: HfApi, repo_for_write: str):
    discussion = api.create_discussion(repo_id=repo_for_write, title="Comment test")
    result = cli(f'hf discussions comment {repo_for_write} {discussion.num} --body "A comment"')
    assert result.exit_code == 0, result.output
    assert f"Commented on #{discussion.num}" in result.output


def test_comment_body_file(api: HfApi, repo_for_write: str, tmp_path):
    discussion = api.create_discussion(repo_id=repo_for_write, title="Comment file test")
    body_file = tmp_path / "comment.md"
    body_file.write_text("Comment from file")
    result = cli(f"hf discussions comment {repo_for_write} {discussion.num} --body-file {body_file}")
    assert result.exit_code == 0, result.output


def test_comment_no_body():
    result = cli("hf discussions comment user/repo 1")
    assert result.exit_code != 0


# =============================================================================
# Close / Reopen
# =============================================================================


def test_close_discussion(api: HfApi, repo_for_write: str):
    discussion = api.create_discussion(repo_id=repo_for_write, title="Close test")
    result = cli(f"hf discussions close {repo_for_write} {discussion.num} --yes")
    assert result.exit_code == 0, result.output
    assert f"Closed #{discussion.num}" in result.output

    details = api.get_discussion_details(repo_id=repo_for_write, discussion_num=discussion.num)
    assert details.status == "closed"


def test_close_with_comment(api: HfApi, repo_for_write: str):
    discussion = api.create_discussion(repo_id=repo_for_write, title="Close with comment")
    result = cli(f'hf discussions close {repo_for_write} {discussion.num} --yes --comment "Done"')
    assert result.exit_code == 0, result.output


def test_close_requires_confirmation(api: HfApi, repo_for_write: str):
    discussion = api.create_discussion(repo_id=repo_for_write, title="Close confirm test")
    result = cli(f"hf discussions close {repo_for_write} {discussion.num}", input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output

    details = api.get_discussion_details(repo_id=repo_for_write, discussion_num=discussion.num)
    assert details.status == "open"


def test_reopen_discussion(api: HfApi, repo_for_write: str):
    discussion = api.create_discussion(repo_id=repo_for_write, title="Reopen test")
    api.change_discussion_status(repo_id=repo_for_write, discussion_num=discussion.num, new_status="closed")

    result = cli(f"hf discussions reopen {repo_for_write} {discussion.num} --yes")
    assert result.exit_code == 0, result.output
    assert f"Reopened #{discussion.num}" in result.output

    details = api.get_discussion_details(repo_id=repo_for_write, discussion_num=discussion.num)
    assert details.status == "open"


# =============================================================================
# Rename
# =============================================================================


def test_rename_discussion(api: HfApi, repo_for_write: str):
    discussion = api.create_discussion(repo_id=repo_for_write, title="Old title")
    result = cli(f'hf discussions rename {repo_for_write} {discussion.num} "New title"')
    assert result.exit_code == 0, result.output
    assert "Renamed" in result.output

    details = api.get_discussion_details(repo_id=repo_for_write, discussion_num=discussion.num)
    assert details.title == "New title"


# =============================================================================
# Merge
# =============================================================================


def test_merge_pr(api: HfApi, repo_for_write: str):
    commit = api.upload_file(
        repo_id=repo_for_write,
        path_or_fileobj=b"merge test",
        path_in_repo="merge_test.txt",
        create_pr=True,
        commit_message="Merge test PR",
    )
    pr_num = int(commit.pr_url.split("/")[-1])
    result = cli(f"hf discussions merge {repo_for_write} {pr_num} --yes")
    assert result.exit_code == 0, result.output
    assert f"Merged #{pr_num}" in result.output


def test_merge_requires_confirmation(api: HfApi, repo_for_write: str):
    commit = api.upload_file(
        repo_id=repo_for_write,
        path_or_fileobj=b"confirm test",
        path_in_repo="confirm_test.txt",
        create_pr=True,
        commit_message="Merge confirm test PR",
    )
    pr_num = int(commit.pr_url.split("/")[-1])
    result = cli(f"hf discussions merge {repo_for_write} {pr_num}", input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output


# =============================================================================
# Diff
# =============================================================================


def test_diff_pr(repo_with_discussion: tuple):
    repo_id, _, pr_num = repo_with_discussion
    result = cli(f"hf discussions diff {repo_id} {pr_num}")
    assert result.exit_code == 0
