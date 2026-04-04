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

import os
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub._buckets import (
    ImportStats,
    _format_import_size,
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


class TestFormatImportSize:
    def test_bytes(self):
        assert _format_import_size(500) == "500 B"

    def test_kb(self):
        assert _format_import_size(1500) == "1.5 KB"

    def test_mb(self):
        assert _format_import_size(1_500_000) == "1.5 MB"

    def test_gb(self):
        assert _format_import_size(1_500_000_000) == "1.5 GB"

    def test_zero(self):
        assert _format_import_size(0) == "0 B"


class TestNormalizeS3Path:
    def test_with_prefix(self):
        assert _normalize_s3_path("s3://my-bucket/prefix") == "my-bucket/prefix"

    def test_without_prefix(self):
        assert _normalize_s3_path("my-bucket/prefix") == "my-bucket/prefix"

    def test_just_bucket(self):
        assert _normalize_s3_path("s3://my-bucket") == "my-bucket"


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

    def test_dry_run_no_upload(self, capsys):
        api = MagicMock()

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = [
            {"name": "my-bucket/file1.txt", "type": "file", "size": 100},
            {"name": "my-bucket/file2.txt", "type": "file", "size": 200},
        ]

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                dry_run=True,
            )

        assert stats.files_transferred == 0
        assert stats.files_skipped == 2
        api.batch_bucket_files.assert_not_called()

        output = capsys.readouterr().out
        assert "file1.txt" in output
        assert "file2.txt" in output
        assert "dry run" in output

    def test_import_with_include_filter(self, capsys):
        api = MagicMock()

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = [
            {"name": "my-bucket/data.parquet", "type": "file", "size": 1000},
            {"name": "my-bucket/readme.md", "type": "file", "size": 50},
            {"name": "my-bucket/other.parquet", "type": "file", "size": 2000},
        ]

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                include=["*.parquet"],
                dry_run=True,
            )

        assert stats.files_skipped == 2
        output = capsys.readouterr().out
        assert "data.parquet" in output
        assert "other.parquet" in output
        assert "readme.md" not in output

    def test_import_with_exclude_filter(self, capsys):
        api = MagicMock()

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = [
            {"name": "my-bucket/data.parquet", "type": "file", "size": 1000},
            {"name": "my-bucket/temp.tmp", "type": "file", "size": 50},
        ]

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                exclude=["*.tmp"],
                dry_run=True,
            )

        assert stats.files_skipped == 1
        output = capsys.readouterr().out
        assert "data.parquet" in output
        assert "temp.tmp" not in output

    def test_import_empty_source(self, capsys):
        api = MagicMock()

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = []

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
            )

        assert stats.files_transferred == 0
        output = capsys.readouterr().out
        assert "No files found" in output

    def test_import_with_dest_prefix(self, capsys):
        api = MagicMock()

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = [
            {"name": "my-bucket/file1.txt", "type": "file", "size": 100},
        ]

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket/imported-data",
                api=api,
                dry_run=True,
            )

        output = capsys.readouterr().out
        assert "imported-data/file1.txt" in output

    def test_actual_transfer(self):
        """Test the full transfer path with mocked S3 and HfApi."""
        api = MagicMock()

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = [
            {"name": "my-bucket/small.txt", "type": "file", "size": 5},
        ]

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("hello")

        mock_s3.get = mock_get

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                quiet=True,
            )

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

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = [
            {"name": "my-bucket/data/train/part-0.parquet", "type": "file", "size": 100},
        ]

        def mock_get(s3_key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("x" * 100)

        mock_s3.get = mock_get

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket/data/",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                quiet=True,
            )

        assert stats.files_transferred == 1
        call_kwargs = api.batch_bucket_files.call_args
        add_list = call_kwargs[1]["add"]
        assert add_list[0][1] == "train/part-0.parquet"

    def test_download_failure_counts(self):
        """Test that S3 download failures are tracked properly."""
        api = MagicMock()

        mock_s3 = MagicMock()
        mock_s3.ls.return_value = [
            {"name": "my-bucket/good.txt", "type": "file", "size": 5},
            {"name": "my-bucket/bad.txt", "type": "file", "size": 5},
        ]

        def mock_get(s3_key, local_path):
            if "bad.txt" in s3_key:
                raise Exception("S3 download error")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("hello")

        mock_s3.get = mock_get

        mock_s3fs_class = MagicMock(return_value=mock_s3)
        mock_s3fs_module = MagicMock()
        mock_s3fs_module.S3FileSystem = mock_s3fs_class

        with patch.dict("sys.modules", {"s3fs": mock_s3fs_module}):
            stats = import_from_s3(
                s3_source="s3://my-bucket",
                bucket_dest="hf://buckets/user/my-bucket",
                api=api,
                quiet=True,
            )

        assert stats.files_transferred == 1
        assert stats.files_failed == 1
