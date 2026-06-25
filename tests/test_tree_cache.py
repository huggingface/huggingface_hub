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
from unittest.mock import patch

import pytest

from huggingface_hub._tree_cache import (
    _IN_MEMORY_TREE_CACHE,
    TREE_CACHE_FORMAT_VERSION,
    TreeCacheEntry,
    read_tree_cache,
    tree_cache_folder_for_local_dir,
    write_tree_cache,
)
from huggingface_hub.file_download import (
    _get_metadata_or_catch_error,
    _xet_file_metadata_from_tree_cache,
    hf_hub_url,
    repo_folder_name,
)
from huggingface_hub.utils._xet import XetTokenType, xet_connection_info_refresh_url


COMMIT_HASH = "0123456789abcdef0123456789abcdef01234567"


def _entries():
    return {
        "config.json": TreeCacheEntry(size=5, blob_id="blob-config"),
        "model.safetensors": TreeCacheEntry(
            size=42,
            blob_id="blob-model",
            lfs_sha256="sha256-model",
            lfs_size=1024,
            xet_hash="xet-model",
        ),
    }


class TestTreeCacheReadWrite:
    def test_round_trip(self, tmp_path: Path):
        write_tree_cache(str(tmp_path), COMMIT_HASH, _entries())
        read = read_tree_cache(str(tmp_path), COMMIT_HASH)
        assert read == _entries()

    def test_file_is_human_readable_and_sorted(self, tmp_path: Path):
        write_tree_cache(str(tmp_path), COMMIT_HASH, _entries())
        path = tmp_path / "trees" / f"{COMMIT_HASH}.json"
        data = json.loads(path.read_text())
        assert data["format_version"] == TREE_CACHE_FORMAT_VERSION
        assert list(data["files"]) == ["config.json", "model.safetensors"]  # sorted by path
        assert data["files"]["config.json"] == {"size": 5, "blob_id": "blob-config"}

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert read_tree_cache(str(tmp_path), COMMIT_HASH) is None

    def test_unknown_format_version_returns_none(self, tmp_path: Path):
        path = tmp_path / "trees" / f"{COMMIT_HASH}.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"format_version": 999, "files": {}}))
        assert read_tree_cache(str(tmp_path), COMMIT_HASH) is None

    def test_corrupted_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "trees" / f"{COMMIT_HASH}.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ not valid json")
        assert read_tree_cache(str(tmp_path), COMMIT_HASH) is None

    def test_in_memory_cache_memoizes_first_read(self, tmp_path: Path):
        write_tree_cache(str(tmp_path), COMMIT_HASH, _entries())
        # Drop the in-memory entry to force a first disk read, then check the result is memoized.
        _IN_MEMORY_TREE_CACHE.pop(str(tmp_path / "trees" / f"{COMMIT_HASH}.json"), None)
        read_tree_cache(str(tmp_path), COMMIT_HASH)
        with patch("huggingface_hub._tree_cache._read_tree_cache_from_disk") as mock_read:
            assert read_tree_cache(str(tmp_path), COMMIT_HASH) == _entries()
            mock_read.assert_not_called()


@pytest.fixture
def tree_cache_folder(tmp_path: Path):
    """Populate and returns a folder with a cached tree listing."""
    storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
    write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())
    return str(storage_folder)


class TestTreeCacheSkipsHeadCall:
    """The download path rebuilds metadata from the cached tree for Xet files, skipping the per-file HEAD call.

    The optimization is intentionally limited to Xet files (when Xet is enabled): that's the only case where
    skipping the HEAD pays off. Regular files always HEAD, even with a cached tree.
    """

    def test_xet_file_metadata_from_tree_cache(self, tree_cache_folder: str):
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            result = _xet_file_metadata_from_tree_cache(
                tree_cache_folder=str(tree_cache_folder),
                repo_id="user/repo",
                repo_type="model",
                commit_hash=COMMIT_HASH,
                filename="model.safetensors",
                endpoint=None,
            )
        assert result is not None
        location, etag, commit_hash, size, xet_file_data, error = result
        assert location == hf_hub_url("user/repo", "model.safetensors", repo_type="model", revision=COMMIT_HASH)
        assert etag == "sha256-model"  # LFS sha256
        assert commit_hash == COMMIT_HASH
        assert size == 1024  # LFS size
        assert error is None
        assert xet_file_data is not None
        assert xet_file_data.file_hash == "xet-model"
        assert xet_file_data.refresh_route == xet_connection_info_refresh_url(
            token_type=XetTokenType.READ, repo_id="user/repo", repo_type="model", revision=COMMIT_HASH
        )

    def test_non_xet_file_returns_none(self, tree_cache_folder: str):
        # `config.json` is a regular (non-Xet) file => the cache is not used, the caller must HEAD.
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            assert (
                _xet_file_metadata_from_tree_cache(
                    tree_cache_folder=tree_cache_folder,
                    repo_id="user/repo",
                    repo_type="model",
                    commit_hash=COMMIT_HASH,
                    filename="config.json",
                    endpoint=None,
                )
                is None
            )

    def test_xet_file_returns_none_when_xet_unavailable(self, tree_cache_folder: str):
        # Even a Xet file falls back to the HEAD call when Xet is not enabled.
        with patch("huggingface_hub.file_download.is_xet_available", return_value=False):
            assert (
                _xet_file_metadata_from_tree_cache(
                    tree_cache_folder=tree_cache_folder,
                    repo_id="user/repo",
                    repo_type="model",
                    commit_hash=COMMIT_HASH,
                    filename="model.safetensors",
                    endpoint=None,
                )
                is None
            )

    def test_no_tree_cache_returns_none(self, tmp_path: Path):  # not populated
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            assert (
                _xet_file_metadata_from_tree_cache(
                    tree_cache_folder=str(storage_folder),
                    repo_id="user/repo",
                    repo_type="model",
                    commit_hash=COMMIT_HASH,
                    filename="model.safetensors",
                    endpoint=None,
                )
                is None
            )

    def test_get_metadata_skips_head_for_xet_file_at_commit_hash(self, tree_cache_folder: str):
        # `get_hf_file_metadata` would do the network HEAD call. With a cached tree for a Xet file it must not.
        with (
            patch("huggingface_hub.file_download.is_xet_available", return_value=True),
            patch("huggingface_hub.file_download.get_hf_file_metadata") as mock_head,
        ):
            _url, etag, commit_hash, size, xet_file_data, error = _get_metadata_or_catch_error(
                repo_id="user/repo",
                filename="model.safetensors",
                repo_type="model",
                revision=COMMIT_HASH,
                endpoint=None,
                etag_timeout=10,
                headers={},
                token=None,
                local_files_only=False,
                tree_cache_folder=tree_cache_folder,
            )
            mock_head.assert_not_called()
        assert etag == "sha256-model"
        assert commit_hash == COMMIT_HASH
        assert size == 1024
        assert xet_file_data is not None
        assert error is None

    def test_get_metadata_heads_for_non_xet_file(self, tree_cache_folder: str):
        # A regular file is not served from the tree cache => the HEAD call must still happen.
        with (
            patch("huggingface_hub.file_download.is_xet_available", return_value=True),
            patch("huggingface_hub.file_download.get_hf_file_metadata", side_effect=RuntimeError("HEAD called")),
        ):
            with pytest.raises(RuntimeError, match="HEAD called"):
                _get_metadata_or_catch_error(
                    repo_id="user/repo",
                    filename="config.json",
                    repo_type="model",
                    revision=COMMIT_HASH,
                    endpoint=None,
                    etag_timeout=10,
                    headers={},
                    token=None,
                    local_files_only=False,
                    tree_cache_folder=tree_cache_folder,
                )

    def test_get_metadata_does_not_use_tree_cache_for_branch(self, tree_cache_folder: str):
        # A branch/tag could have moved since the listing was cached => the HEAD call must still happen.
        with patch("huggingface_hub.file_download.get_hf_file_metadata", side_effect=RuntimeError("HEAD called")):
            with pytest.raises(RuntimeError, match="HEAD called"):
                _get_metadata_or_catch_error(
                    repo_id="user/repo",
                    filename="config.json",
                    repo_type="model",
                    revision="main",  # not a commit hash
                    endpoint=None,
                    etag_timeout=10,
                    headers={},
                    token=None,
                    local_files_only=False,
                    tree_cache_folder=tree_cache_folder,
                )


class TestTreeCacheForLocalDir:
    def test_xet_file_metadata_from_tree_cache_reads_local_dir_location(self, tmp_path: Path):
        # Metadata under .cache/huggingface/trees/...
        folder = tree_cache_folder_for_local_dir(str(tmp_path))
        assert folder == str(tmp_path / ".cache" / "huggingface")
        write_tree_cache(folder, COMMIT_HASH, _entries())

        # Cache is read correctly
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            result = _xet_file_metadata_from_tree_cache(
                tree_cache_folder=folder,
                repo_id="user/repo",
                repo_type="model",
                commit_hash=COMMIT_HASH,
                filename="model.safetensors",
                endpoint=None,
            )
        assert result is not None
        _location, etag, _commit_hash, _size, xet_file_data, _error = result
        assert etag == "sha256-model"
        assert xet_file_data is not None
