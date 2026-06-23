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
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import snapshot_download
from huggingface_hub._tree_cache import (
    _IN_MEMORY_TREE_CACHE,
    TREE_CACHE_FORMAT_VERSION,
    TreeCacheEntry,
    read_tree_cache,
    tree_cache_folder_for_local_dir,
    write_tree_cache,
)
from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError
from huggingface_hub.file_download import (
    _file_metadata_from_tree_cache,
    _get_metadata_or_catch_error,
    hf_hub_url,
    repo_folder_name,
)
from huggingface_hub.utils._xet import XetTokenType, xet_connection_info_refresh_url


COMMIT_HASH = "0123456789abcdef0123456789abcdef01234567"


def _entries():
    return {
        "config.json": TreeCacheEntry(path="config.json", size=5, blob_id="blob-config"),
        "model.safetensors": TreeCacheEntry(
            path="model.safetensors",
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


class TestTreeCacheSkipsHeadCall:
    """The download path rebuilds metadata from the cached tree for Xet files, skipping the per-file HEAD call.

    The optimization is intentionally limited to Xet files (when Xet is enabled): that's the only case where
    skipping the HEAD pays off. Regular files always HEAD, even with a cached tree.
    """

    def test_file_metadata_from_tree_cache(self, tmp_path: Path):
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())

        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            result = _file_metadata_from_tree_cache(
                tree_cache_folder=str(storage_folder),
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

    def test_non_xet_file_returns_none(self, tmp_path: Path):
        # `config.json` is a regular (non-Xet) file => the cache is not used, the caller must HEAD.
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            assert (
                _file_metadata_from_tree_cache(
                    tree_cache_folder=str(storage_folder),
                    repo_id="user/repo",
                    repo_type="model",
                    commit_hash=COMMIT_HASH,
                    filename="config.json",
                    endpoint=None,
                )
                is None
            )

    def test_xet_file_returns_none_when_xet_unavailable(self, tmp_path: Path):
        # Even a Xet file falls back to the HEAD call when Xet is not enabled.
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())
        with patch("huggingface_hub.file_download.is_xet_available", return_value=False):
            assert (
                _file_metadata_from_tree_cache(
                    tree_cache_folder=str(storage_folder),
                    repo_id="user/repo",
                    repo_type="model",
                    commit_hash=COMMIT_HASH,
                    filename="model.safetensors",
                    endpoint=None,
                )
                is None
            )

    def test_no_tree_cache_returns_none(self, tmp_path: Path):
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            assert (
                _file_metadata_from_tree_cache(
                    tree_cache_folder=str(storage_folder),
                    repo_id="user/repo",
                    repo_type="model",
                    commit_hash=COMMIT_HASH,
                    filename="model.safetensors",
                    endpoint=None,
                )
                is None
            )

    def test_get_metadata_skips_head_for_xet_file_at_commit_hash(self, tmp_path: Path):
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())

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
                tree_cache_folder=str(storage_folder),
            )
            mock_head.assert_not_called()
        assert etag == "sha256-model"
        assert commit_hash == COMMIT_HASH
        assert size == 1024
        assert xet_file_data is not None
        assert error is None

    def test_get_metadata_heads_for_non_xet_file(self, tmp_path: Path):
        # A regular file is not served from the tree cache => the HEAD call must still happen.
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())

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
                    tree_cache_folder=str(storage_folder),
                )

    def test_get_metadata_does_not_use_tree_cache_for_branch(self, tmp_path: Path):
        # A branch/tag could have moved since the listing was cached => the HEAD call must still happen.
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())

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
                    tree_cache_folder=str(storage_folder),
                )


class TestTreeCacheForLocalDir:
    """`local_dir` downloads cache the tree listing under `.cache/huggingface/`, never at the local_dir root.

    Repo files are written directly at the root of `local_dir`, so a `trees/` folder there would collide with a
    repo file literally named `trees/...`. The cache must live under the reserved metadata dir instead.
    """

    def test_tree_cache_folder_is_under_cache_huggingface(self, tmp_path: Path):
        folder = tree_cache_folder_for_local_dir(str(tmp_path))
        assert folder == str(tmp_path / ".cache" / "huggingface")

    def test_tree_cache_does_not_collide_with_repo_file_named_trees(self, tmp_path: Path):
        # Simulate a repo that contains a file literally named `trees/<something>` at its root.
        repo_tree_file = tmp_path / "trees" / "collides.json"
        repo_tree_file.parent.mkdir(parents=True)
        repo_tree_file.write_text("I am a repo file, not the cache")

        # Writing the tree cache must not touch that repo file.
        folder = tree_cache_folder_for_local_dir(str(tmp_path))
        write_tree_cache(folder, COMMIT_HASH, _entries())

        # The repo file is untouched...
        assert repo_tree_file.read_text() == "I am a repo file, not the cache"
        # ...and the cache lives separately, under `.cache/huggingface/trees/`.
        cache_file = tmp_path / ".cache" / "huggingface" / "trees" / f"{COMMIT_HASH}.json"
        assert cache_file.is_file()
        assert read_tree_cache(folder, COMMIT_HASH) == _entries()

    def test_file_metadata_from_tree_cache_reads_local_dir_location(self, tmp_path: Path):
        # The metadata-rebuild path must find a tree written under the local_dir metadata folder.
        folder = tree_cache_folder_for_local_dir(str(tmp_path))
        write_tree_cache(folder, COMMIT_HASH, _entries())
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True):
            result = _file_metadata_from_tree_cache(
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


def _build_cache(cache_dir: Path, present_files: list[str]) -> Path:
    """Create a fake cache layout with a cached tree listing and only `present_files` materialized."""
    storage_folder = cache_dir / repo_folder_name(repo_id="user/repo", repo_type="model")
    snapshot_folder = storage_folder / "snapshots" / COMMIT_HASH
    snapshot_folder.mkdir(parents=True)
    write_tree_cache(str(storage_folder), COMMIT_HASH, _entries())
    for file in present_files:
        file_path = snapshot_folder / file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x")
    return snapshot_folder


class TestIncompleteSnapshotOffline:
    def test_complete_snapshot_is_returned(self, tmp_path: Path):
        snapshot_folder = _build_cache(tmp_path, present_files=["config.json", "model.safetensors"])
        result = snapshot_download("user/repo", cache_dir=str(tmp_path), revision=COMMIT_HASH, local_files_only=True)
        assert os.path.realpath(result) == os.path.realpath(str(snapshot_folder))

    def test_incomplete_snapshot_raises(self, tmp_path: Path):
        _build_cache(tmp_path, present_files=["config.json"])  # model.safetensors missing
        with pytest.raises(IncompleteSnapshotError) as exc_info:
            snapshot_download("user/repo", cache_dir=str(tmp_path), revision=COMMIT_HASH, local_files_only=True)
        assert "model.safetensors" in str(exc_info.value)
        # subclass of LocalEntryNotFoundError / FileNotFoundError so existing handlers keep working
        assert isinstance(exc_info.value, LocalEntryNotFoundError)
        assert isinstance(exc_info.value, FileNotFoundError)

    def test_incomplete_but_filtered_out_is_returned(self, tmp_path: Path):
        # model.safetensors is missing but ignored => snapshot is complete for the requested patterns
        snapshot_folder = _build_cache(tmp_path, present_files=["config.json"])
        result = snapshot_download(
            "user/repo",
            cache_dir=str(tmp_path),
            revision=COMMIT_HASH,
            local_files_only=True,
            ignore_patterns=["*.safetensors"],
        )
        assert os.path.realpath(result) == os.path.realpath(str(snapshot_folder))

    def test_without_cached_tree_legacy_behavior_returns_folder(self, tmp_path: Path):
        # No tree cache => we cannot tell completeness => return the (possibly partial) folder as before.
        storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
        snapshot_folder = storage_folder / "snapshots" / COMMIT_HASH
        snapshot_folder.mkdir(parents=True)
        (snapshot_folder / "config.json").write_text("x")  # model.safetensors missing, but no tree to know
        result = snapshot_download("user/repo", cache_dir=str(tmp_path), revision=COMMIT_HASH, local_files_only=True)
        assert os.path.realpath(result) == os.path.realpath(str(snapshot_folder))


def _build_local_dir(local_dir: Path, present_files: list[str]) -> Path:
    """Materialize a `local_dir` with a cached tree listing (under `.cache/huggingface/`) and only some files."""
    write_tree_cache(tree_cache_folder_for_local_dir(str(local_dir)), COMMIT_HASH, _entries())
    for file in present_files:
        file_path = local_dir / file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x")
    return local_dir


class TestIncompleteSnapshotOfflineLocalDir:
    """Same completeness guarantees as the cache_dir path, but for `local_dir` downloads."""

    def test_complete_local_dir_is_returned(self, tmp_path: Path):
        local_dir = tmp_path / "out"
        local_dir.mkdir()
        _build_local_dir(local_dir, present_files=["config.json", "model.safetensors"])
        result = snapshot_download("user/repo", local_dir=str(local_dir), revision=COMMIT_HASH, local_files_only=True)
        assert os.path.realpath(result) == os.path.realpath(str(local_dir))

    def test_incomplete_local_dir_raises(self, tmp_path: Path):
        local_dir = tmp_path / "out"
        local_dir.mkdir()
        _build_local_dir(local_dir, present_files=["config.json"])  # model.safetensors missing
        with pytest.raises(IncompleteSnapshotError) as exc_info:
            snapshot_download("user/repo", local_dir=str(local_dir), revision=COMMIT_HASH, local_files_only=True)
        assert "model.safetensors" in str(exc_info.value)
        assert isinstance(exc_info.value, LocalEntryNotFoundError)

    def test_incomplete_local_dir_but_filtered_out_is_returned(self, tmp_path: Path):
        local_dir = tmp_path / "out"
        local_dir.mkdir()
        _build_local_dir(local_dir, present_files=["config.json"])  # model.safetensors missing but ignored
        result = snapshot_download(
            "user/repo",
            local_dir=str(local_dir),
            revision=COMMIT_HASH,
            local_files_only=True,
            ignore_patterns=["*.safetensors"],
        )
        assert os.path.realpath(result) == os.path.realpath(str(local_dir))

    def test_tree_cache_file_does_not_collide_with_repo_file_named_trees(self, tmp_path: Path):
        # A repo file literally named `trees/...` at the local_dir root must survive a download that writes
        # the tree cache, and must not be mistaken for the cache.
        local_dir = tmp_path / "out"
        local_dir.mkdir()
        repo_tree_file = local_dir / "trees" / "collides.json"
        repo_tree_file.parent.mkdir(parents=True)
        repo_tree_file.write_text("I am a repo file")

        # Cache the tree listing (goes under `.cache/huggingface/trees/`) and mark the snapshot complete.
        _build_local_dir(local_dir, present_files=["config.json", "model.safetensors"])
        # The repo file is untouched and distinct from the cache file.
        assert repo_tree_file.read_text() == "I am a repo file"
        cache_file = local_dir / ".cache" / "huggingface" / "trees" / f"{COMMIT_HASH}.json"
        assert cache_file.is_file()
        assert repo_tree_file != cache_file

        # Offline, the snapshot is reported complete (both requested files are present).
        result = snapshot_download("user/repo", local_dir=str(local_dir), revision=COMMIT_HASH, local_files_only=True)
        assert os.path.realpath(result) == os.path.realpath(str(local_dir))
