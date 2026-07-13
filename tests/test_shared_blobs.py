"""Tests for the cache-wide shared blob store (see `huggingface_hub._shared_blobs`)."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import constants
from huggingface_hub._shared_blobs import (
    are_hardlinks_supported,
    publish_blob_to_shared_store,
    shared_blob_path,
    shared_blobs_dir,
    shared_store_inodes,
    sweep_shared_blobs,
    try_link_from_shared_store,
)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import scan_cache_dir
from huggingface_hub.utils._xet import XetFileData


XET_HASH = "cf" + "ab" * 31
OTHER_XET_HASH = "0d" + "12" * 31
CONTENT = b"shared-content" * 100


def _write_blob(path: Path, content: bytes = CONTENT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _make_store_entry(cache_dir: Path, xet_hash: str, content: bytes = CONTENT) -> Path:
    return _write_blob(shared_blob_path(cache_dir, xet_hash), content)


def _make_cached_file(
    cache_dir: Path,
    repo_folder: str,
    commit: str,
    filename: str,
    etag: str,
    content: bytes = CONTENT,
    *,
    ref: str | None = None,
    link_to: Path | None = None,
) -> Path:
    """Create a repo blob + snapshot pointer, optionally hardlinked to an existing file."""
    repo = cache_dir / repo_folder
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "blobs").mkdir(exist_ok=True)
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    if ref is not None:
        (repo / "refs" / ref).write_text(commit)
    blob = repo / "blobs" / etag
    if not blob.exists():
        if link_to is not None:
            os.link(link_to, blob)
        else:
            blob.write_bytes(content)
    (snapshot / filename).symlink_to(Path("..") / ".." / "blobs" / etag)
    return blob


def test_module_imports_standalone() -> None:
    # Regression test: importing `_shared_blobs` before anything else used to crash with
    # a circular import (`_shared_blobs` -> `utils` -> `_cache_manager` -> back into the
    # partially initialized `_shared_blobs`). A fresh interpreter is required to catch it.
    result = subprocess.run(
        [sys.executable, "-c", "import huggingface_hub._shared_blobs"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestStoreHelpers:
    def test_shared_blob_path_prefix_split(self, tmp_path: Path) -> None:
        assert shared_blob_path(tmp_path, XET_HASH) == tmp_path / "blobs" / "cf" / XET_HASH

    @pytest.mark.parametrize("bad_hash", ["", "abc", "CF" + "AB" * 31, "../" + "a" * 61, "g" * 64])
    def test_shared_blob_path_invalid_hash(self, tmp_path: Path, bad_hash: str) -> None:
        with pytest.raises(ValueError):
            shared_blob_path(tmp_path, bad_hash)

    def test_are_hardlinks_supported(self, tmp_path: Path) -> None:
        assert are_hardlinks_supported(tmp_path)
        assert are_hardlinks_supported(tmp_path)  # memoized second call
        assert list(tmp_path.iterdir()) == []  # probe files cleaned up


class TestLinkAndPublish:
    def test_publish_then_link_roundtrip(self, tmp_path: Path) -> None:
        blob_a = _write_blob(tmp_path / "models--a" / "blobs" / "etag")
        publish_blob_to_shared_store(blob_path=str(blob_a), xet_hash=XET_HASH, cache_dir=tmp_path)

        store_entry = shared_blob_path(tmp_path, XET_HASH)
        assert os.path.samestat(store_entry.stat(), blob_a.stat())
        assert blob_a.stat().st_nlink == 2

        blob_b = tmp_path / "models--b" / "blobs" / "etag"
        blob_b.parent.mkdir(parents=True)
        assert try_link_from_shared_store(
            blob_path=str(blob_b), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        assert blob_b.read_bytes() == CONTENT
        assert blob_b.stat().st_nlink == 3

    def test_link_miss_returns_false(self, tmp_path: Path) -> None:
        blob = tmp_path / "models--a" / "blobs" / "etag"
        blob.parent.mkdir(parents=True)
        assert not try_link_from_shared_store(
            blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        assert not blob.exists()

    def test_link_size_mismatch_evicts_entry(self, tmp_path: Path) -> None:
        store_entry = _make_store_entry(tmp_path, XET_HASH, b"truncated")
        blob = tmp_path / "models--a" / "blobs" / "etag"
        blob.parent.mkdir(parents=True)
        assert not try_link_from_shared_store(
            blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        assert not store_entry.exists()  # corrupted entry evicted from the store

    def test_link_invalid_hash_returns_false(self, tmp_path: Path) -> None:
        assert not try_link_from_shared_store(
            blob_path=str(tmp_path / "blob"), xet_hash="not-a-hash", cache_dir=tmp_path, expected_size=None
        )

    def test_publish_is_idempotent(self, tmp_path: Path) -> None:
        blob = _write_blob(tmp_path / "models--a" / "blobs" / "etag")
        publish_blob_to_shared_store(blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path)
        publish_blob_to_shared_store(blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path)
        assert blob.stat().st_nlink == 2  # same-inode short-circuit, no churn

    def test_publish_replaces_existing_entry_with_verified_copy(self, tmp_path: Path) -> None:
        # An existing same-size entry might be corrupted (indistinguishable without a
        # full re-hash) -> the copy the publisher just verified always takes over the
        # store entry, while other repos hardlinking the old inode are untouched.
        corrupt_content = bytes(len(CONTENT))  # same size, different bytes
        store_entry = _make_store_entry(tmp_path, XET_HASH, corrupt_content)
        other_repo_blob = tmp_path / "models--b" / "blobs" / "etag"
        other_repo_blob.parent.mkdir(parents=True)
        os.link(store_entry, other_repo_blob)

        blob = _write_blob(tmp_path / "models--a" / "blobs" / "etag")
        publish_blob_to_shared_store(blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path)

        assert store_entry.read_bytes() == CONTENT  # store healed
        assert os.path.samestat(store_entry.stat(), blob.stat())
        assert other_repo_blob.read_bytes() == corrupt_content  # old inode untouched
        assert list(blob.parent.iterdir()) == [blob]  # no leftover tmp file

    def test_publish_invalid_hash_is_noop(self, tmp_path: Path) -> None:
        blob = _write_blob(tmp_path / "models--a" / "blobs" / "etag")
        publish_blob_to_shared_store(blob_path=str(blob), xet_hash="not-a-hash", cache_dir=tmp_path)
        assert not shared_blobs_dir(tmp_path).exists()


class TestSweep:
    def test_sweep_removes_orphans_only(self, tmp_path: Path) -> None:
        orphan = _make_store_entry(tmp_path, XET_HASH)
        referenced = _make_store_entry(tmp_path, OTHER_XET_HASH)
        blob = tmp_path / "models--a" / "blobs" / "etag"
        blob.parent.mkdir(parents=True)
        os.link(referenced, blob)

        freed = sweep_shared_blobs(tmp_path)

        assert freed == len(CONTENT)
        assert not orphan.exists()
        assert not orphan.parent.exists()  # empty prefix dir removed
        assert referenced.exists()
        assert blob.read_bytes() == CONTENT

    def test_sweep_no_store(self, tmp_path: Path) -> None:
        assert sweep_shared_blobs(tmp_path) == 0

    def test_shared_store_inodes(self, tmp_path: Path) -> None:
        entry = _make_store_entry(tmp_path, XET_HASH)
        assert shared_store_inodes(tmp_path) == {entry.stat().st_ino}
        assert shared_store_inodes(tmp_path / "no-store") == set()


class TestScanAndDelete:
    def test_scan_skips_store_dir(self, tmp_path: Path) -> None:
        _make_store_entry(tmp_path, XET_HASH)
        _make_cached_file(tmp_path, "models--org--repo", "a" * 40, "file.bin", "11" * 32, ref="main")

        report = scan_cache_dir(tmp_path)

        assert report.warnings == []
        assert {repo.repo_id for repo in report.repos} == {"org/repo"}
        assert report.size_on_disk == len(CONTENT)  # store entry not double-counted
        assert report.cache_dir == tmp_path

    def test_delete_shared_blob_accounting_and_sweep(self, tmp_path: Path) -> None:
        rev_shared, rev_private, rev_b = "a" * 40, "c" * 40, "b" * 40
        private_content = b"private" * 10

        # repoA: rev_shared (main) -> shared blob, rev_private (dev) -> private blob.
        # repoB: rev_b (main) -> same shared blob. Shared blob is hardlinked to the store.
        blob_a = _make_cached_file(tmp_path, "models--org--repoA", rev_shared, "model.bin", "11" * 32, ref="main")
        publish_blob_to_shared_store(blob_path=str(blob_a), xet_hash=XET_HASH, cache_dir=tmp_path)
        _make_cached_file(
            tmp_path, "models--org--repoA", rev_private, "other.bin", "22" * 32, private_content, ref="dev"
        )
        _make_cached_file(tmp_path, "models--org--repoB", rev_b, "model.bin", "11" * 32, ref="main", link_to=blob_a)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        assert store_entry.stat().st_nlink == 3

        # Deleting repoA's shared revision frees nothing: repoB still holds the blob.
        strategy = scan_cache_dir(tmp_path).delete_revisions(rev_shared)
        assert strategy.expected_freed_size == 0
        strategy.execute()
        assert store_entry.stat().st_nlink == 2
        snapshot_b = tmp_path / "models--org--repoB" / "snapshots" / rev_b / "model.bin"
        assert snapshot_b.read_bytes() == CONTENT

        # Deleting repoB (last referrer) frees the blob: the store orphan is swept.
        strategy = scan_cache_dir(tmp_path).delete_revisions(rev_b)
        assert strategy.expected_freed_size == len(CONTENT)
        strategy.execute()
        assert not store_entry.exists()

        # Regular (non-deduplicated) blob accounting is unchanged.
        strategy = scan_cache_dir(tmp_path).delete_revisions(rev_private)
        assert strategy.expected_freed_size == len(private_content)


class TestDownloadIntegration:
    """End-to-end `hf_hub_download` flow with mocked HEAD metadata and Xet download."""

    ETAG = "e7" * 32
    COMMIT = "d" * 40

    @pytest.fixture(autouse=True)
    def enable_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(constants, "HF_HUB_ENABLE_SHARED_BLOBS", True)

    def _download(self, cache_dir: Path, repo_id: str, xet_downloads: list[str], **kwargs) -> Path:
        """Run `hf_hub_download` with a fake HEAD response and a fake `xet_get`."""
        metadata = (
            "https://example.com/file.bin",  # url_to_download
            self.ETAG,
            self.COMMIT,
            len(CONTENT),  # expected_size
            XetFileData(file_hash=XET_HASH, refresh_route="https://example.com/refresh"),
            None,  # head_call_error
        )

        def fake_xet_get(*, incomplete_path: Path, **kwargs) -> None:
            xet_downloads.append(repo_id)
            incomplete_path.write_bytes(CONTENT)

        with (
            patch("huggingface_hub.file_download._get_metadata_or_catch_error", return_value=metadata),
            patch("huggingface_hub.file_download.xet_get", side_effect=fake_xet_get),
            patch("huggingface_hub.file_download.is_xet_available", return_value=True),
        ):
            return Path(hf_hub_download(repo_id, "file.bin", cache_dir=str(cache_dir), **kwargs))

    def test_second_repo_reuses_blob_without_download(self, tmp_path: Path) -> None:
        xet_downloads: list[str] = []

        path_a = self._download(tmp_path, "org/repoA", xet_downloads)
        assert path_a.read_bytes() == CONTENT
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        assert store_entry.stat().st_nlink == 2  # repoA blob + store entry

        path_b = self._download(tmp_path, "org/repoB", xet_downloads)
        assert path_b.read_bytes() == CONTENT
        assert store_entry.stat().st_nlink == 3  # repoB blob hardlinked, not downloaded
        assert xet_downloads == ["org/repoA"]

    def test_force_download_replaces_store_entry(self, tmp_path: Path) -> None:
        xet_downloads: list[str] = []
        self._download(tmp_path, "org/repoA", xet_downloads)
        self._download(tmp_path, "org/repoB", xet_downloads)

        path_a = self._download(tmp_path, "org/repoA", xet_downloads, force_download=True)
        assert xet_downloads == ["org/repoA", "org/repoA"]

        # The fresh verified copy replaced the store entry; repoB keeps its own inode.
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        blob_a = path_a.resolve()
        assert os.path.samestat(store_entry.stat(), blob_a.stat())
        assert store_entry.stat().st_nlink == 2

    def test_http_fallback_does_not_publish(self, tmp_path: Path) -> None:
        metadata = (
            "https://example.com/file.bin",
            self.ETAG,
            self.COMMIT,
            len(CONTENT),
            XetFileData(file_hash=XET_HASH, refresh_route="https://example.com/refresh"),
            None,
        )

        def fake_http_get(url: str, temp_file, **kwargs) -> None:
            temp_file.write(CONTENT)

        with (
            patch("huggingface_hub.file_download._get_metadata_or_catch_error", return_value=metadata),
            patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get),
            patch("huggingface_hub.file_download.is_xet_available", return_value=False),
        ):
            path = Path(hf_hub_download("org/repoA", "file.bin", cache_dir=str(tmp_path)))

        assert path.read_bytes() == CONTENT
        # Unverified content (plain HTTP download) is never published to the store.
        assert not shared_blob_path(tmp_path, XET_HASH).exists()

    def test_store_disabled_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(constants, "HF_HUB_ENABLE_SHARED_BLOBS", False)
        xet_downloads: list[str] = []
        self._download(tmp_path, "org/repoA", xet_downloads)
        assert not shared_blobs_dir(tmp_path).exists()

    def test_degraded_no_symlink_mode_disables_store(self, tmp_path: Path) -> None:
        # In the no-symlink layout, blobs are moved into snapshots/ where users edit
        # files in place: sharing inodes there would propagate edits across repos.
        xet_downloads: list[str] = []
        with patch("huggingface_hub.file_download.are_symlinks_supported", return_value=False):
            path = self._download(tmp_path, "org/repoA", xet_downloads)
        assert path.read_bytes() == CONTENT
        assert not shared_blobs_dir(tmp_path).exists()
