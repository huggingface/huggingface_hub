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
"""Tests for S3-to-HF bucket import functionality."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub._buckets import (
    ImportStats,
    _format_size,
    _normalize_s3_path,
    import_from_s3,
)


class TestImportStats:
    def test_throughput_calculation(self):
        stats = ImportStats(bytes_transferred=10 * 1024 * 1024, elapsed_seconds=2.0)
        assert abs(stats.throughput_mb_s - 5.0) < 0.01

    def test_throughput_zero_elapsed(self):
        stats = ImportStats(bytes_transferred=1000, elapsed_seconds=0.0)
        assert stats.throughput_mb_s == 0.0

    def test_summary_str(self):
        stats = ImportStats(
            files_transferred=10,
            files_skipped=2,
            files_failed=1,
            bytes_transferred=1024 * 1024,
            elapsed_seconds=5.0,
        )
        s = stats.summary_str()
        assert "10 file(s)" in s
        assert "skipped 2" in s
        assert "failed 1" in s
        assert "5.0s" in s


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500, human_readable=True) == "500 B"

    def test_kb(self):
        assert _format_size(1500, human_readable=True) == "1.5 KB"

    def test_mb(self):
        assert _format_size(1_500_000, human_readable=True) == "1.5 MB"

    def test_gb(self):
        assert _format_size(1_500_000_000, human_readable=True) == "1.5 GB"

    def test_zero(self):
        assert _format_size(0, human_readable=True) == "0 B"

    def test_not_human_readable(self):
        assert _format_size(1500) == "1500"


class TestNormalizeS3Path:
    def test_with_prefix(self):
        assert _normalize_s3_path("s3://my-bucket/prefix") == "my-bucket/prefix"

    def test_without_prefix(self):
        assert _normalize_s3_path("my-bucket/prefix") == "my-bucket/prefix"

    def test_just_bucket(self):
        assert _normalize_s3_path("s3://my-bucket") == "my-bucket"


@pytest.fixture(autouse=True)
def _mock_xet_preflight():
    """Mock the xet-write-token preflight so tests with a MagicMock api don't hit the network."""
    with patch("huggingface_hub._buckets.fetch_xet_connection_info_from_repo_info") as m:
        yield m


def _make_s3_mocks(files):
    """Helper to create mocked s3fs module and filesystem.

    Args:
        files: list of dicts with "name", "type", and "size" keys.

    Returns:
        tuple: (mock_s3fs_module, mock_s3_filesystem)
    """
    mock_s3 = MagicMock()
    mock_s3.ls.return_value = files
    mock_s3fs_class = MagicMock(return_value=mock_s3)
    mock_s3fs_module = MagicMock()
    mock_s3fs_module.S3FileSystem = mock_s3fs_class
    return mock_s3fs_module, mock_s3


class TestImportFromS3:
    def test_invalid_source_raises(self):
        api = MagicMock()
        with pytest.raises(ValueError, match="S3 URI"):
            import_from_s3(
                s3_source="/local/path",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
            )

    def test_invalid_dest_raises(self):
        api = MagicMock()
        with pytest.raises(ValueError, match="bucket path"):
            import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="/local/path",
                api=api,
            )

    def test_missing_s3fs_raises(self):
        api = MagicMock()
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "s3fs":
                raise ImportError("No module named 's3fs'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="s3fs"):
                import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                )

    def test_dry_run_outputs_jsonl(self, capsys):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks(
            [
                {"name": "my-bucket/file1.txt", "type": "file", "size": 100},
                {"name": "my-bucket/file2.txt", "type": "file", "size": 200},
            ]
        )

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            result = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                dry_run=True,
            )

        # Should return a SyncPlan, not ImportStats
        from huggingface_hub._buckets import SyncPlan

        assert isinstance(result, SyncPlan)

        # Should output JSONL to stdout
        output = capsys.readouterr().out
        lines = [line for line in output.strip().split("\n") if line]
        header = json.loads(lines[0])
        assert header["type"] == "header"
        assert header["source"] == "s3://my-bucket"

        ops = [json.loads(line) for line in lines[1:]]
        assert len(ops) == 2
        assert all(op["action"] == "upload" for op in ops)

        api.batch_bucket_files.assert_not_called()

    def test_import_with_include_filter(self, capsys):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks(
            [
                {"name": "my-bucket/data.parquet", "type": "file", "size": 1000},
                {"name": "my-bucket/readme.md", "type": "file", "size": 50},
                {"name": "my-bucket/other.parquet", "type": "file", "size": 2000},
            ]
        )

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            result = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                include=["*.parquet"],
                dry_run=True,
            )

        from huggingface_hub._buckets import SyncPlan

        assert isinstance(result, SyncPlan)
        assert len(result.operations) == 2
        paths = [op.path for op in result.operations]
        assert "data.parquet" in paths
        assert "other.parquet" in paths

    def test_import_with_exclude_filter(self, capsys):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks(
            [
                {"name": "my-bucket/data.parquet", "type": "file", "size": 1000},
                {"name": "my-bucket/temp.tmp", "type": "file", "size": 50},
            ]
        )

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            result = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                exclude=["*.tmp"],
                dry_run=True,
            )

        from huggingface_hub._buckets import SyncPlan

        assert isinstance(result, SyncPlan)
        assert len(result.operations) == 1
        assert result.operations[0].path == "data.parquet"

    def test_import_with_filter_from(self, capsys):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks(
            [
                {"name": "my-bucket/data.parquet", "type": "file", "size": 1000},
                {"name": "my-bucket/temp.tmp", "type": "file", "size": 50},
                {"name": "my-bucket/readme.md", "type": "file", "size": 30},
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("- *.tmp\n")
            f.write("- *.md\n")
            filter_file = f.name

        try:
            with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
                result = import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                    filter_from=filter_file,
                    dry_run=True,
                )

            from huggingface_hub._buckets import SyncPlan

            assert isinstance(result, SyncPlan)
            assert len(result.operations) == 1
            assert result.operations[0].path == "data.parquet"
        finally:
            os.unlink(filter_file)

    def test_import_empty_source(self, capsys):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks([])

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            result = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
            )

        # Empty source returns empty ImportStats (no plan to execute)
        assert isinstance(result, ImportStats)
        assert result.files_transferred == 0

    def test_import_with_dest_prefix(self, capsys):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks(
            [
                {"name": "my-bucket/file1.txt", "type": "file", "size": 100},
            ]
        )

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            result = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket/imported-data",
                api=api,
                dry_run=True,
            )

        from huggingface_hub._buckets import SyncPlan

        assert isinstance(result, SyncPlan)
        assert len(result.operations) == 1
        assert result.operations[0].path == "file1.txt"

    def test_actual_transfer(self):
        """Test the full transfer path with mocked S3 and HfApi."""
        api = MagicMock()
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [
                {"name": "my-bucket/small.txt", "type": "file", "size": 5},
            ]
        )

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("hello")

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                quiet=True,
            )

        assert isinstance(stats, ImportStats)
        assert stats.files_transferred == 1
        assert stats.bytes_transferred == 5
        assert stats.files_failed == 0
        api.batch_bucket_files.assert_called_once()
        call_kwargs = api.batch_bucket_files.call_args
        assert call_kwargs[0][0] == "user/my-bucket"
        add_list = call_kwargs[1]["add"]
        assert len(add_list) == 1
        assert add_list[0][1] == "small.txt"

    def test_transfer_with_s3_prefix(self):
        """Test that S3 prefix paths are handled correctly."""
        api = MagicMock()
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [
                {"name": "my-bucket/data/train/part-0.parquet", "type": "file", "size": 100},
            ]
        )

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("x" * 100)

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket/data/",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                quiet=True,
            )

        assert isinstance(stats, ImportStats)
        assert stats.files_transferred == 1
        call_kwargs = api.batch_bucket_files.call_args
        add_list = call_kwargs[1]["add"]
        assert add_list[0][1] == "train/part-0.parquet"

    def test_download_failure_counts(self):
        """Test that S3 download failures are tracked properly."""
        api = MagicMock()
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [
                {"name": "my-bucket/good.txt", "type": "file", "size": 5},
                {"name": "my-bucket/bad.txt", "type": "file", "size": 5},
            ]
        )

        def mock_get(s3_key, local_path):
            if "bad.txt" in s3_key:
                raise Exception("S3 download error")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("hello")

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                quiet=True,
            )

        assert isinstance(stats, ImportStats)
        assert stats.files_transferred == 1
        assert stats.files_failed == 1


class TestImportPlanApply:
    def test_plan_saves_file(self):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks(
            [
                {"name": "my-bucket/file1.txt", "type": "file", "size": 100},
                {"name": "my-bucket/file2.txt", "type": "file", "size": 200},
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            plan_file = f.name

        try:
            with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
                result = import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                    plan=plan_file,
                )

            from huggingface_hub._buckets import SyncPlan

            assert isinstance(result, SyncPlan)

            with open(plan_file) as f:
                lines = f.readlines()
            header = json.loads(lines[0])
            assert header["type"] == "header"
            assert header["source"] == "s3://my-bucket"
            assert header["dest"] == "hf://buckets/user/my-bucket"

            ops = [json.loads(line) for line in lines[1:]]
            assert len(ops) == 2
            assert all(op["action"] == "upload" for op in ops)

            api.batch_bucket_files.assert_not_called()
        finally:
            os.unlink(plan_file)

    def test_apply_executes_plan(self):
        api = MagicMock()

        # Write a plan file
        plan_content = [
            json.dumps(
                {
                    "type": "header",
                    "source": "s3://my-bucket",
                    "dest": "hf://buckets/user/my-bucket",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "summary": {"uploads": 1, "downloads": 0, "deletes": 0, "skips": 0, "total_size": 5},
                }
            ),
            json.dumps(
                {
                    "type": "operation",
                    "action": "upload",
                    "path": "small.txt",
                    "size": 5,
                    "reason": "new file",
                }
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(plan_content) + "\n")
            plan_file = f.name

        mock_s3 = MagicMock()

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("hello")

        mock_s3.get = mock_get
        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        try:
            with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
                stats = import_from_s3(
                    api=api,
                    apply=plan_file,
                    quiet=True,
                )

            assert isinstance(stats, ImportStats)
            assert stats.files_transferred == 1
            api.batch_bucket_files.assert_called_once()
        finally:
            os.unlink(plan_file)

    def test_apply_validation_errors(self):
        api = MagicMock()

        with pytest.raises(ValueError, match="Cannot specify source/dest"):
            import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                apply="plan.jsonl",
            )

        with pytest.raises(ValueError, match="Cannot specify both --plan and --apply"):
            import_from_s3(api=api, apply="plan.jsonl", plan="other.jsonl")

        with pytest.raises(ValueError, match="Cannot specify --include"):
            import_from_s3(api=api, apply="plan.jsonl", include=["*.txt"])

        with pytest.raises(ValueError, match="Cannot specify --exclude"):
            import_from_s3(api=api, apply="plan.jsonl", exclude=["*.tmp"])

        with pytest.raises(ValueError, match="Cannot specify --filter-from"):
            import_from_s3(api=api, apply="plan.jsonl", filter_from="filters.txt")

        with pytest.raises(ValueError, match="Cannot specify --dry-run"):
            import_from_s3(api=api, apply="plan.jsonl", dry_run=True)

    def test_dry_run_and_plan_conflict(self):
        api = MagicMock()
        mock_s3fs_module, _ = _make_s3_mocks([])

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            with pytest.raises(ValueError, match="Cannot specify both --dry-run and --plan"):
                import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                    dry_run=True,
                    plan="out.jsonl",
                )

    def test_missing_source_raises(self):
        api = MagicMock()
        with pytest.raises(ValueError, match="Source S3 URI is required"):
            import_from_s3(bucket_dest="hf://buckets/user/my-bucket", api=api)

    def test_missing_dest_raises(self):
        api = MagicMock()
        with pytest.raises(ValueError, match="Destination bucket path is required"):
            import_from_s3(s3_source="s3://my-bucket", api=api)


class TestBufferSize:
    def test_single_file_exceeds_buffer(self):
        api = MagicMock()
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [
                {"name": "my-bucket/huge.bin", "type": "file", "size": 10_000_000_000},
            ]
        )

        def mock_get(s3_key, local_path):
            pass  # Should never be called

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            with pytest.raises(ValueError, match="exceeds --buffer-size"):
                import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                    buffer_size=1_000_000_000,  # 1GB
                    quiet=True,
                )

    def test_buffer_size_splits_batches(self):
        """Verify that buffer_size constrains batch sizes."""
        api = MagicMock()
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [
                {"name": "my-bucket/a.bin", "type": "file", "size": 600},
                {"name": "my-bucket/b.bin", "type": "file", "size": 600},
                {"name": "my-bucket/c.bin", "type": "file", "size": 600},
            ]
        )

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("x")

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                buffer_size=1000,  # Only fits 1 file per batch (600 < 1000, but 600+600 > 1000)
                quiet=True,
            )

        assert isinstance(stats, ImportStats)
        assert stats.files_transferred == 3
        # With buffer_size=1000, each 600-byte file fits alone but two don't.
        # So we expect 3 batches of 1 file each = 3 batch_bucket_files calls.
        assert api.batch_bucket_files.call_count == 3


class TestAuthFailureFastFail:
    """Verify that auth failures don't trigger recursive split-and-retry against /xet-write-token."""

    def test_preflight_fails_fast_on_401(self, _mock_xet_preflight):
        """A 401 from the write-token preflight should surface immediately, before any S3 download."""
        from huggingface_hub.errors import HfHubHTTPError, XetAuthorizationError

        mock_response = MagicMock()
        mock_response.status_code = 401
        _mock_xet_preflight.side_effect = HfHubHTTPError("unauthorized", response=mock_response)

        api = MagicMock()
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [
                {"name": "my-bucket/f1.txt", "type": "file", "size": 10},
                {"name": "my-bucket/f2.txt", "type": "file", "size": 10},
            ]
        )
        mock_s3.get = MagicMock()

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            with pytest.raises(XetAuthorizationError):
                import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                    quiet=True,
                )

        # No S3 downloads, no batch_bucket_files calls — we failed before the batch loop.
        mock_s3.get.assert_not_called()
        api.batch_bucket_files.assert_not_called()

    def test_batch_auth_error_is_not_split_and_retried(self, _mock_xet_preflight):
        """A XetAuthorizationError raised by batch_bucket_files must re-raise, not split+retry."""
        from huggingface_hub.errors import XetAuthorizationError

        api = MagicMock()
        api.batch_bucket_files.side_effect = XetAuthorizationError("nope")
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [{"name": f"my-bucket/f{i}.txt", "type": "file", "size": 10} for i in range(8)]
        )

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("x")

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            with pytest.raises(XetAuthorizationError):
                import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                    quiet=True,
                )

        # The old buggy behavior would split the batch of 8 into 4+4, then 2+2+2+2, then 1×8,
        # producing 15 calls. We must stop at the first one.
        assert api.batch_bucket_files.call_count == 1

    def test_batch_401_http_error_is_not_split_and_retried(self, _mock_xet_preflight):
        """A HfHubHTTPError(401) from batch_bucket_files must also re-raise, not split+retry."""
        from huggingface_hub.errors import HfHubHTTPError

        mock_response = MagicMock()
        mock_response.status_code = 401

        api = MagicMock()
        api.batch_bucket_files.side_effect = HfHubHTTPError("unauthorized", response=mock_response)
        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [{"name": f"my-bucket/f{i}.txt", "type": "file", "size": 10} for i in range(4)]
        )

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("x")

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            with pytest.raises(HfHubHTTPError):
                import_from_s3(
                    s3_source="s3://my-bucket",
                    bucket_dest="hf://buckets/user/my-bucket",
                    api=api,
                    quiet=True,
                )

        assert api.batch_bucket_files.call_count == 1

    def test_non_auth_batch_error_still_splits(self, _mock_xet_preflight):
        """Non-auth failures (e.g. bad file) should still split-and-retry to isolate the bad file."""
        api = MagicMock()
        # Fail the first call (full batch) with a generic error, succeed on sub-batches.
        api.batch_bucket_files.side_effect = [RuntimeError("boom"), None, None]

        mock_s3fs_module, mock_s3 = _make_s3_mocks(
            [{"name": f"my-bucket/f{i}.txt", "type": "file", "size": 10} for i in range(2)]
        )

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("x")

        mock_s3.get = mock_get

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                quiet=True,
            )

        # Generic error: 1 (full batch, fails) + 2 (each file as its own chunk) = 3 calls.
        assert api.batch_bucket_files.call_count == 3
