# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Unit tests for the `hf download` CLI command, including `--stdout` streaming."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from huggingface_hub import constants
from huggingface_hub.cli.download import _flush_stdout, _write_stdout_bytes
from huggingface_hub.cli.hf import app
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import SoftTemporaryDirectory

from .testing_constants import DUMMY_MODEL_ID


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestDownloadStdout:
    def test_download_stdout_cached_file(self, runner: CliRunner) -> None:
        with SoftTemporaryDirectory() as tmpdir:
            cached_file = Path(tmpdir) / "config.json"
            cached_file.write_bytes(b"cached binary data 12345")
            with (
                patch(
                    "huggingface_hub.cli.download.try_to_load_from_cache", return_value=str(cached_file)
                ) as mock_cache,
                patch("huggingface_hub.cli.download.http_stream_backoff") as mock_stream,
            ):
                result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout"])

        assert result.exit_code == 0
        assert result.stdout_bytes == b"cached binary data 12345"
        mock_cache.assert_called_once()
        mock_stream.assert_not_called()

    def test_download_stdout_remote_file(self, runner: CliRunner) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = [b"chunk1_", b"chunk2"]
        mock_cm = Mock()
        mock_cm.__enter__ = Mock(return_value=mock_response)
        mock_cm.__exit__ = Mock(return_value=None)

        with (
            patch("huggingface_hub.cli.download.try_to_load_from_cache", return_value=None) as mock_cache,
            patch("huggingface_hub.cli.download.http_stream_backoff", return_value=mock_cm) as mock_stream,
            patch("huggingface_hub.cli.download.hf_raise_for_status") as mock_raise,
        ):
            result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout"])

        assert result.exit_code == 0
        assert result.stdout_bytes == b"chunk1_chunk2"
        mock_cache.assert_called_once()
        mock_stream.assert_called_once()
        assert mock_stream.call_args.kwargs.get("timeout") == constants.HF_HUB_DOWNLOAD_TIMEOUT
        mock_raise.assert_called_once_with(mock_response)

    def test_download_stdout_force_download_bypasses_cache(self, runner: CliRunner) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = [b"fresh_content"]
        mock_cm = Mock()
        mock_cm.__enter__ = Mock(return_value=mock_response)
        mock_cm.__exit__ = Mock(return_value=None)

        with (
            patch("huggingface_hub.cli.download.try_to_load_from_cache") as mock_cache,
            patch("huggingface_hub.cli.download.http_stream_backoff", return_value=mock_cm) as mock_stream,
        ):
            result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout", "--force-download"])

        assert result.exit_code == 0
        assert result.stdout_bytes == b"fresh_content"
        mock_cache.assert_not_called()
        mock_stream.assert_called_once()

    def test_download_stdout_hf_uri(self, runner: CliRunner) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = [b"uri_data"]
        mock_cm = Mock()
        mock_cm.__enter__ = Mock(return_value=mock_response)
        mock_cm.__exit__ = Mock(return_value=None)

        with (
            patch("huggingface_hub.cli.download.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.cli.download.http_stream_backoff", return_value=mock_cm) as mock_stream,
        ):
            result = runner.invoke(
                app, ["download", "hf://datasets/author/dataset@refs/pr/3/data/train.csv", "--stdout"]
            )

        assert result.exit_code == 0
        assert result.stdout_bytes == b"uri_data"
        assert "datasets/author/dataset/resolve/refs%2Fpr%2F3/data/train.csv" in mock_stream.call_args[0][1]

    def test_download_stdout_rejects_multiple_files(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "file1.txt", "file2.txt", "--stdout"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "`--stdout` can only be used when downloading a single file." in str(result.exception)

    def test_download_stdout_rejects_no_files(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "--stdout"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "`--stdout` can only be used when downloading a single file." in str(result.exception)

    def test_download_stdout_rejects_subfolder(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "art/", "--stdout"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "`--stdout` can only be used when downloading a single file." in str(result.exception)

    def test_download_stdout_rejects_wildcard(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "*.json", "--stdout"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "`--stdout` can only be used when downloading a single file." in str(result.exception)

    def test_download_stdout_rejects_bracket_wildcard(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "model-[0-9].bin", "--stdout"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "`--stdout` can only be used when downloading a single file." in str(result.exception)

    def test_download_stdout_rejects_path_traversal(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "../secret.txt", "--stdout"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "cannot contain a '..' path segment" in str(result.exception)

    def test_download_stdout_rejects_hf_uri_subfolder(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", "hf://datasets/author/dataset/data/", "--stdout"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "`--stdout` can only be used when downloading a single file." in str(result.exception)

    def test_download_stdout_rejects_local_dir(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout", "--local-dir", "./dir"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "Cannot use both `--stdout` and `--local-dir`" in str(result.exception)

    def test_download_stdout_rejects_dry_run(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout", "--dry-run"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "Cannot use both `--stdout` and `--dry-run`" in str(result.exception)

    def test_download_stdout_rejects_include(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout", "--include", "*.json"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "Cannot use both `--stdout` and `--include`" in str(result.exception)

    def test_download_stdout_rejects_exclude(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout", "--exclude", "*.bin"])
        assert result.exit_code != 0
        assert isinstance(result.exception, CLIError)
        assert "Cannot use both `--stdout` and `--exclude`" in str(result.exception)

    def test_download_stdout_broken_pipe_handled_cleanly(self, runner: CliRunner) -> None:
        with SoftTemporaryDirectory() as tmpdir:
            cached_file = Path(tmpdir) / "config.json"
            cached_file.write_bytes(b"cached content")
            with (
                patch("huggingface_hub.cli.download.try_to_load_from_cache", return_value=str(cached_file)),
                patch("huggingface_hub.cli.download._write_stdout_bytes", side_effect=BrokenPipeError("Broken pipe")),
            ):
                result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout"])

        assert result.exit_code == 0
        assert result.exception is None

    def test_download_stdout_keyboard_interrupt_handled_cleanly(self, runner: CliRunner) -> None:
        with SoftTemporaryDirectory() as tmpdir:
            cached_file = Path(tmpdir) / "config.json"
            cached_file.write_bytes(b"cached content")
            with (
                patch("huggingface_hub.cli.download.try_to_load_from_cache", return_value=str(cached_file)),
                patch("huggingface_hub.cli.download._write_stdout_bytes", side_effect=KeyboardInterrupt()),
            ):
                result = runner.invoke(app, ["download", DUMMY_MODEL_ID, "config.json", "--stdout"])

        assert result.exit_code == 0
        assert result.exception is None

    def test_download_stdout_none_stdout_safety(self) -> None:
        with patch.object(sys, "stdout", None):
            _write_stdout_bytes(b"some bytes")
            _flush_stdout()
