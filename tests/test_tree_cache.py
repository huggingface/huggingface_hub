"""Network-free tests for the on-disk tree listing cache and the resulting `snapshot_download` behavior.

The integration tests in `test_snapshot_download.py` exercise the online path against the Hub. Here we
only build cache folders by hand and check the offline / `local_files_only` logic, so these tests need
neither network access nor a token.
"""

import json
import os
from pathlib import Path

import pytest

from huggingface_hub import snapshot_download
from huggingface_hub._tree_cache import (
    TREE_CACHE_FORMAT_VERSION,
    TreeCacheEntry,
    read_tree_cache,
    write_tree_cache,
)
from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError
from huggingface_hub.file_download import repo_folder_name


COMMIT_HASH = "0123456789abcdef0123456789abcdef01234567"  # valid-looking 40-char commit hash


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


class TestTreeCacheEntry:
    def test_etag_and_size_for_git_file(self):
        entry = TreeCacheEntry(path="config.json", size=5, blob_id="blob-config")
        assert entry.etag == "blob-config"  # git blob id
        assert entry.file_size == 5

    def test_etag_and_size_for_lfs_file(self):
        entry = TreeCacheEntry(
            path="model.safetensors", size=42, blob_id="blob-model", lfs_sha256="sha256-model", lfs_size=1024
        )
        assert entry.etag == "sha256-model"  # LFS sha256, matches `/resolve` ETag
        assert entry.file_size == 1024


class TestTreeCacheReadWrite:
    def test_round_trip(self, tmp_path: Path):
        write_tree_cache(str(tmp_path), COMMIT_HASH, "user/repo", "model", _entries())
        read = read_tree_cache(str(tmp_path), COMMIT_HASH)
        assert read == _entries()

    def test_file_is_human_readable_and_sorted(self, tmp_path: Path):
        write_tree_cache(str(tmp_path), COMMIT_HASH, "user/repo", "model", _entries())
        path = tmp_path / "trees" / f"{COMMIT_HASH}.json"
        data = json.loads(path.read_text())
        assert data["format_version"] == TREE_CACHE_FORMAT_VERSION
        assert data["repo_id"] == "user/repo"
        assert data["commit_hash"] == COMMIT_HASH
        assert list(data["files"]) == ["config.json", "model.safetensors"]  # sorted by path
        # git-only file has no lfs/xet keys
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


def _build_cache(cache_dir: Path, present_files: list[str]) -> Path:
    """Create a fake cache layout with a cached tree listing and only `present_files` materialized."""
    storage_folder = cache_dir / repo_folder_name(repo_id="user/repo", repo_type="model")
    snapshot_folder = storage_folder / "snapshots" / COMMIT_HASH
    snapshot_folder.mkdir(parents=True)
    write_tree_cache(str(storage_folder), COMMIT_HASH, "user/repo", "model", _entries())
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
