# Copyright 2024-present, the HuggingFace Inc. team.
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
"""Unit tests for OCI Object Storage integration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from huggingface_hub.oci_file_system import (
    _get_oci_fs,
    _hf_hub_download_to_oci,
    _snapshot_download_to_oci,
    is_oci_uri,
    upload_file_to_oci,
    upload_folder_to_oci,
)


# ---------------------------------------------------------------------------
# is_oci_uri
# ---------------------------------------------------------------------------


class TestIsOciUri:
    def test_valid_uri(self):
        assert is_oci_uri("oci://my-bucket@namespace/path/to/file") is True

    def test_valid_uri_no_path(self):
        assert is_oci_uri("oci://my-bucket@namespace") is True

    def test_local_path(self):
        assert is_oci_uri("/local/path") is False

    def test_s3_uri(self):
        assert is_oci_uri("s3://bucket/key") is False

    def test_none(self):
        assert is_oci_uri(None) is False

    def test_empty_string(self):
        assert is_oci_uri("") is False

    def test_path_object(self):
        # Path objects are not strings so should return False
        assert is_oci_uri(Path("/some/path")) is False


# ---------------------------------------------------------------------------
# _get_oci_fs
# ---------------------------------------------------------------------------


class TestGetOciFs:
    def test_raises_when_ocifs_not_installed(self):
        with patch("huggingface_hub.oci_file_system.is_ocifs_available", return_value=False):
            with pytest.raises(ImportError, match="ocifs"):
                _get_oci_fs()

    def test_returns_filesystem_when_available(self):
        mock_fs = MagicMock()
        mock_ocifs = MagicMock()
        mock_ocifs.OCIFileSystem.return_value = mock_fs

        with patch("huggingface_hub.oci_file_system.is_ocifs_available", return_value=True):
            with patch.dict("sys.modules", {"ocifs": mock_ocifs}):
                fs = _get_oci_fs()

        assert fs is mock_fs
        mock_ocifs.OCIFileSystem.assert_called_once_with()

    def test_passes_storage_options(self):
        mock_fs = MagicMock()
        mock_ocifs = MagicMock()
        mock_ocifs.OCIFileSystem.return_value = mock_fs
        opts = {"config": "/path/to/config", "profile": "DEFAULT"}

        with patch("huggingface_hub.oci_file_system.is_ocifs_available", return_value=True):
            with patch.dict("sys.modules", {"ocifs": mock_ocifs}):
                _get_oci_fs(opts)

        mock_ocifs.OCIFileSystem.assert_called_once_with(**opts)


# ---------------------------------------------------------------------------
# upload_file_to_oci
# ---------------------------------------------------------------------------


class TestUploadFileToOci:
    def test_raises_on_invalid_uri(self):
        with pytest.raises(ValueError, match="oci://"):
            upload_file_to_oci("/local/file.txt", "s3://wrong-bucket/file.txt")

    def test_appends_filename_when_uri_ends_with_slash(self):
        mock_fs = MagicMock()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            local_path = f.name
        try:
            with patch("huggingface_hub.oci_file_system._get_oci_fs", return_value=mock_fs):
                dest = upload_file_to_oci(local_path, "oci://bucket@ns/models/bert/")
            filename = Path(local_path).name
            assert dest == f"oci://bucket@ns/models/bert/{filename}"
            mock_fs.put_file.assert_called_once_with(local_path, dest)
        finally:
            os.unlink(local_path)

    def test_uses_oci_uri_directly_when_no_trailing_slash(self):
        mock_fs = MagicMock()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            local_path = f.name
        try:
            with patch("huggingface_hub.oci_file_system._get_oci_fs", return_value=mock_fs):
                dest = upload_file_to_oci(local_path, "oci://bucket@ns/models/bert/config.json")
            assert dest == "oci://bucket@ns/models/bert/config.json"
            mock_fs.put_file.assert_called_once_with(local_path, dest)
        finally:
            os.unlink(local_path)


# ---------------------------------------------------------------------------
# upload_folder_to_oci
# ---------------------------------------------------------------------------


class TestUploadFolderToOci:
    def test_raises_on_invalid_uri(self):
        with pytest.raises(ValueError, match="oci://"):
            upload_folder_to_oci("/local/folder", "s3://wrong-bucket/")

    def test_uploads_all_files_with_relative_paths(self):
        mock_fs = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a small directory tree
            (Path(tmp_dir) / "subdir").mkdir()
            (Path(tmp_dir) / "config.json").write_text("{}")
            (Path(tmp_dir) / "subdir" / "model.safetensors").write_bytes(b"\x00" * 8)

            with patch("huggingface_hub.oci_file_system._get_oci_fs", return_value=mock_fs):
                result = upload_folder_to_oci(tmp_dir, "oci://bucket@ns/models/bert")

        assert result == "oci://bucket@ns/models/bert"
        put_calls = {c.args[1] for c in mock_fs.put_file.call_args_list}
        assert "oci://bucket@ns/models/bert/config.json" in put_calls
        assert "oci://bucket@ns/models/bert/subdir/model.safetensors" in put_calls

    def test_delete_local_removes_folder(self):
        mock_fs = MagicMock()
        with tempfile.TemporaryDirectory() as parent:
            folder = Path(parent) / "snapshot"
            folder.mkdir()
            (folder / "file.txt").write_text("hello")

            with patch("huggingface_hub.oci_file_system._get_oci_fs", return_value=mock_fs):
                upload_folder_to_oci(str(folder), "oci://bucket@ns/dest", delete_local=True)

            assert not folder.exists()

    def test_delete_local_false_keeps_folder(self):
        mock_fs = MagicMock()
        with tempfile.TemporaryDirectory() as parent:
            folder = Path(parent) / "snapshot"
            folder.mkdir()
            (folder / "file.txt").write_text("hello")

            with patch("huggingface_hub.oci_file_system._get_oci_fs", return_value=mock_fs):
                upload_folder_to_oci(str(folder), "oci://bucket@ns/dest", delete_local=False)

            assert folder.exists()


# ---------------------------------------------------------------------------
# _snapshot_download_to_oci
# ---------------------------------------------------------------------------


class TestSnapshotDownloadToOci:
    def test_downloads_to_temp_then_uploads(self):
        """_snapshot_download_to_oci downloads to a temp dir then uploads to OCI."""
        captured = {}

        def fake_snapshot_download(repo_id, local_dir=None, **kwargs):
            assert os.path.isdir(local_dir)
            captured["tmp_dir"] = local_dir
            (Path(local_dir) / "config.json").write_text("{}")
            return local_dir

        mock_fs = MagicMock()
        with (
            patch("huggingface_hub.oci_file_system._get_oci_fs", return_value=mock_fs),
            # Patch snapshot_download inside the lazy import in _snapshot_download_to_oci
            patch("huggingface_hub._snapshot_download.snapshot_download", fake_snapshot_download),
        ):
            result = _snapshot_download_to_oci(
                "bert-base-uncased",
                oci_uri="oci://bucket@ns/models/bert",
            )

        assert result == "oci://bucket@ns/models/bert"
        # Temp dir cleaned up after context exit
        assert not os.path.exists(captured["tmp_dir"])

    def test_returns_oci_uri(self):
        """Return value is always the destination oci_uri."""
        mock_fs = MagicMock()
        with (
            patch("huggingface_hub.oci_file_system._get_oci_fs", return_value=mock_fs),
            patch("huggingface_hub._snapshot_download.snapshot_download", return_value="/tmp/fake"),
        ):
            result = _snapshot_download_to_oci("org/repo", oci_uri="oci://b@ns/dest")

        assert result == "oci://b@ns/dest"


# ---------------------------------------------------------------------------
# Integration: snapshot_download routes to OCI when local_dir is oci://
# ---------------------------------------------------------------------------


class TestSnapshotDownloadOciRouting:
    def test_routes_to_oci_when_local_dir_is_oci_uri(self):
        """snapshot_download delegates to _snapshot_download_to_oci for oci:// local_dir."""
        from huggingface_hub._snapshot_download import snapshot_download

        # _snapshot_download_to_oci is imported lazily inside the function;
        # patch it at the source module so the lazy import picks up the mock.
        with patch(
            "huggingface_hub.oci_file_system._snapshot_download_to_oci",
            return_value="oci://bucket@ns/models/bert",
        ) as mock_oci:
            result = snapshot_download(
                "bert-base-uncased",
                local_dir="oci://bucket@ns/models/bert",
            )

        mock_oci.assert_called_once()
        assert result == "oci://bucket@ns/models/bert"
