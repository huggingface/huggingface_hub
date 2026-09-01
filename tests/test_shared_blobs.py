"""Tests for the cache-wide shared blob store (see `huggingface_hub._shared_blobs`)."""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import constants
from huggingface_hub._shared_blobs import (
    _relative_blob_path,
    is_shared_blobs_dir,
    publish_blob_to_shared_store,
    shared_blob_manifest_path,
    shared_blob_path,
    shared_blob_target,
    shared_blobs_dir,
    sweep_shared_blob,
    try_link_from_shared_store,
)
from huggingface_hub.file_download import _chmod_and_move, hf_hub_download
from huggingface_hub.utils import scan_cache_dir
from huggingface_hub.utils._xet import XetFileData


XET_HASH = "cf" + "ab" * 31
OTHER_XET_HASH = "0d" + "12" * 31
CONTENT = b"shared-content" * 100

requires_reliable_symlinks = pytest.mark.skipif(
    os.name == "nt", reason="symlink syscalls are flaky on Windows CI runners"
)


def _write_blob(path: Path, content: bytes = CONTENT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _publish(
    cache_dir: Path,
    *,
    repo_folder: str = "models--org--repoA",
    etag: str = "etag",
    xet_hash: str = XET_HASH,
    content: bytes = CONTENT,
    replace_existing: bool = False,
) -> Path:
    blob = _write_blob(cache_dir / repo_folder / "blobs" / etag, content)
    assert publish_blob_to_shared_store(
        blob_path=str(blob),
        xet_hash=xet_hash,
        cache_dir=cache_dir,
        expected_size=len(content),
        replace_existing=replace_existing,
    )
    return blob


def _make_cached_file(
    cache_dir: Path,
    repo_folder: str,
    commit: str,
    filename: str,
    etag: str,
    content: bytes = CONTENT,
    *,
    ref: str | None = None,
) -> Path:
    repo = cache_dir / repo_folder
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    blob = _write_blob(repo / "blobs" / etag, content)
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    if ref is not None:
        (repo / "refs" / ref).write_text(commit)
    (snapshot / filename).symlink_to(Path("..") / ".." / "blobs" / etag)
    return blob


def test_module_imports_standalone() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import huggingface_hub._shared_blobs"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestStoreHelpers:
    def test_shared_blob_path_prefix_split(self, tmp_path: Path) -> None:
        assert shared_blob_path(tmp_path, XET_HASH) == tmp_path / "blobs" / "cf" / XET_HASH
        assert shared_blob_manifest_path(tmp_path, XET_HASH) == tmp_path / "blobs" / "cf" / f"{XET_HASH}.refs"

    def test_relative_blob_path_accepts_windows_extended_length_prefix(self, tmp_path: Path) -> None:
        blob = tmp_path / "models--org--repo" / "blobs" / "etag"

        assert _relative_blob_path(f"\\\\?\\{blob}", tmp_path) == "models--org--repo/blobs/etag"

    @pytest.mark.parametrize("bad_hash", ["", "abc", "CF" + "AB" * 31, "../" + "a" * 61, "g" * 64])
    def test_shared_blob_path_invalid_hash(self, tmp_path: Path, bad_hash: str) -> None:
        with pytest.raises(ValueError):
            shared_blob_path(tmp_path, bad_hash)

    def test_unmarked_store_is_not_adopted(self, tmp_path: Path) -> None:
        unowned_file = _write_blob(shared_blobs_dir(tmp_path) / "cf" / "user-file")
        blob = _write_blob(tmp_path / "models--org--repo" / "blobs" / "etag")

        assert not publish_blob_to_shared_store(
            blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        assert unowned_file.read_bytes() == CONTENT
        assert blob.is_file() and not blob.is_symlink()
        assert not is_shared_blobs_dir(shared_blobs_dir(tmp_path))

    def test_abandoned_marker_temp_does_not_disable_store(self, tmp_path: Path) -> None:
        store_dir = shared_blobs_dir(tmp_path)
        store_dir.mkdir()
        marker_temp = store_dir / ".huggingface-shared-blobs.deadbeef.tmp"
        marker_temp.write_text("1\n")

        blob = _publish(tmp_path)

        assert blob.read_bytes() == CONTENT
        assert is_shared_blobs_dir(store_dir)
        assert not marker_temp.exists()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are required")
    def test_existing_empty_store_recovers_shared_mode(self, tmp_path: Path) -> None:
        tmp_path.chmod(0o2770)
        store_dir = shared_blobs_dir(tmp_path)
        store_dir.mkdir(mode=0o700)
        store_dir.chmod(0o700)

        _publish(tmp_path)

        assert store_dir.stat().st_mode & 0o7777 == 0o2770

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are required")
    def test_existing_empty_prefix_recovers_shared_mode(self, tmp_path: Path) -> None:
        tmp_path.chmod(0o2770)
        _publish(tmp_path)
        prefix_dir = shared_blob_path(tmp_path, OTHER_XET_HASH).parent
        prefix_dir.mkdir(mode=0o700)
        prefix_dir.chmod(0o700)

        _publish(tmp_path, repo_folder="models--org--repoB", xet_hash=OTHER_XET_HASH)

        assert prefix_dir.stat().st_mode & 0o7777 == 0o2770

    def test_symlinked_prefix_is_refused(self, tmp_path: Path) -> None:
        first_blob = _publish(tmp_path)
        first_blob.unlink()
        prefix = shared_blob_path(tmp_path, OTHER_XET_HASH).parent
        external = tmp_path / "external"
        external.mkdir()
        prefix.symlink_to(external, target_is_directory=True)
        blob = _write_blob(tmp_path / "models--org--repoB" / "blobs" / "etag")

        assert not publish_blob_to_shared_store(
            blob_path=str(blob), xet_hash=OTHER_XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        assert list(external.iterdir()) == []

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are required")
    def test_store_permissions_follow_shared_cache_root(self, tmp_path: Path) -> None:
        tmp_path.chmod(0o2770)
        blob = _publish(tmp_path)

        store_entry = shared_blob_path(tmp_path, XET_HASH)
        assert store_entry.parent.parent.stat().st_mode & 0o7777 == 0o2770
        assert store_entry.parent.stat().st_mode & 0o7777 == 0o2770
        assert store_entry.stat().st_mode & 0o777 == 0o440
        assert store_entry.stat().st_gid == store_entry.parent.stat().st_gid
        assert shared_blob_manifest_path(tmp_path, XET_HASH).stat().st_mode & 0o777 == 0o666
        assert blob.is_symlink()

    @requires_reliable_symlinks
    def test_lock_falls_back_when_flock_is_unsupported(self, tmp_path: Path) -> None:
        with patch("huggingface_hub._shared_blobs.FileLock.acquire", side_effect=NotImplementedError):
            blob = _publish(tmp_path)

        assert blob.read_bytes() == CONTENT
        assert not list(shared_blob_path(tmp_path, XET_HASH).parent.glob("*.soft"))

    @requires_reliable_symlinks
    def test_stale_soft_lock_fails_without_blocking(self, tmp_path: Path) -> None:
        _publish(tmp_path)
        store_path = shared_blob_path(tmp_path, XET_HASH)
        Path(f"{store_path}.lock.soft").touch()
        blob = _write_blob(tmp_path / "models--org--repoB" / "blobs" / "etag")

        with (
            patch("huggingface_hub._shared_blobs.FileLock.acquire", side_effect=NotImplementedError),
            patch("huggingface_hub._shared_blobs._SOFT_LOCK_TIMEOUT", 0),
        ):
            assert not publish_blob_to_shared_store(
                blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
            )

        assert blob.is_file() and not blob.is_symlink()
        assert blob.read_bytes() == CONTENT


@requires_reliable_symlinks
class TestLinkAndPublish:
    def test_publish_then_link_roundtrip(self, tmp_path: Path) -> None:
        blob_a = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)

        assert store_entry.is_file() and not store_entry.is_symlink()
        assert blob_a.is_symlink()
        assert shared_blob_target(blob_a, tmp_path) == store_entry
        assert blob_a.read_bytes() == CONTENT
        assert store_entry.stat().st_nlink == 1

        blob_b = tmp_path / "models--org--repoB" / "blobs" / "etag"
        blob_b.parent.mkdir(parents=True)
        assert try_link_from_shared_store(
            blob_path=str(blob_b), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        assert blob_b.is_symlink()
        assert blob_b.read_bytes() == CONTENT
        assert shared_blob_manifest_path(tmp_path, XET_HASH).read_text().splitlines() == [
            "models--org--repoA/blobs/etag",
            "models--org--repoB/blobs/etag",
        ]

    def test_manifest_is_flushed_before_repo_symlink_is_replaced(self, tmp_path: Path) -> None:
        blob = _write_blob(tmp_path / "models--org--repoA" / "blobs" / "etag")
        events: list[str] = []

        from huggingface_hub import _shared_blobs

        original_append = _shared_blobs._append_manifest_reference
        original_replace = _shared_blobs.os.replace

        def append(*args, **kwargs):
            events.append("manifest")
            return original_append(*args, **kwargs)

        def replace(src, dst):
            if Path(dst) == shared_blob_path(tmp_path, XET_HASH):
                events.append("store")
            if Path(dst) == blob and Path(src).name.endswith(".shared"):
                events.append("symlink")
            return original_replace(src, dst)

        with (
            patch("huggingface_hub._shared_blobs._append_manifest_reference", side_effect=append),
            patch("huggingface_hub._shared_blobs.os.replace", side_effect=replace),
        ):
            assert publish_blob_to_shared_store(
                blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
            )

        assert events == ["manifest", "store", "symlink"]

    def test_size_mismatch_is_ignored_not_evicted(self, tmp_path: Path) -> None:
        blob_a = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        blob_a.unlink()
        store_entry.chmod(0o600)
        store_entry.write_bytes(b"truncated")
        blob_b = tmp_path / "models--org--repoB" / "blobs" / "etag"
        blob_b.parent.mkdir(parents=True)

        assert not try_link_from_shared_store(
            blob_path=str(blob_b), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        assert store_entry.read_bytes() == b"truncated"
        assert not blob_b.exists()

    def test_force_publication_atomically_replaces_canonical_file(self, tmp_path: Path) -> None:
        blob_a = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        old_inode = store_entry.stat().st_ino
        fresh_content = CONTENT[::-1]
        fresh_blob = _write_blob(tmp_path / "models--org--repoB" / "blobs" / "etag", fresh_content)

        assert publish_blob_to_shared_store(
            blob_path=str(fresh_blob),
            xet_hash=XET_HASH,
            cache_dir=tmp_path,
            expected_size=len(fresh_content),
            replace_existing=True,
        )

        assert store_entry.stat().st_ino != old_inode
        assert blob_a.read_bytes() == fresh_content
        assert fresh_blob.read_bytes() == fresh_content
        assert store_entry.stat().st_nlink == 1

    def test_manifest_failure_keeps_regular_repo_blob(self, tmp_path: Path) -> None:
        blob = _write_blob(tmp_path / "models--org--repoA" / "blobs" / "etag")
        with patch(
            "huggingface_hub._shared_blobs._append_manifest_reference",
            side_effect=PermissionError("read-only manifest"),
        ):
            assert not publish_blob_to_shared_store(
                blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
            )

        assert blob.is_file() and not blob.is_symlink()
        assert blob.read_bytes() == CONTENT

    def test_failed_publish_raises_when_regular_repo_blob_cannot_be_restored(self, tmp_path: Path) -> None:
        blob = _write_blob(tmp_path / "models--org--repoA" / "blobs" / "etag")

        from huggingface_hub import _shared_blobs

        original_replace = _shared_blobs.os.replace

        def fail_repo_symlink_replace(src, dst):
            if Path(dst) == blob and Path(src).name.endswith(".shared"):
                raise PermissionError("cannot publish repo symlink")
            return original_replace(src, dst)

        with (
            patch("huggingface_hub._shared_blobs.os.replace", side_effect=fail_repo_symlink_replace),
            patch("huggingface_hub._shared_blobs.shutil.copyfile", side_effect=PermissionError("cannot restore blob")),
            pytest.raises(OSError, match="Could not restore repo blob"),
        ):
            publish_blob_to_shared_store(
                blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
            )

        assert not os.path.lexists(blob)
        assert shared_blob_path(tmp_path, XET_HASH).read_bytes() == CONTENT


@requires_reliable_symlinks
class TestManifestGc:
    def test_sweep_removes_stale_manifest_entries_and_orphan(self, tmp_path: Path) -> None:
        blob_a = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        blob_b = tmp_path / "models--org--repoB" / "blobs" / "etag"
        blob_b.parent.mkdir(parents=True)
        assert try_link_from_shared_store(
            blob_path=str(blob_b), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )

        blob_a.unlink()
        assert sweep_shared_blob(store_entry, cache_dir=tmp_path) == 0
        assert shared_blob_manifest_path(tmp_path, XET_HASH).read_text() == "models--org--repoB/blobs/etag\n"

        blob_b.unlink()
        assert sweep_shared_blob(store_entry, cache_dir=tmp_path) == len(CONTENT)
        assert not store_entry.exists()
        assert not shared_blob_manifest_path(tmp_path, XET_HASH).exists()

    def test_missing_manifest_leaks_instead_of_deleting(self, tmp_path: Path) -> None:
        blob = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        blob.unlink()
        shared_blob_manifest_path(tmp_path, XET_HASH).unlink()

        assert sweep_shared_blob(store_entry, cache_dir=tmp_path) == 0
        assert store_entry.exists()

    def test_manifest_lines_are_validated(self, tmp_path: Path) -> None:
        blob = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        blob.unlink()
        shared_blob_manifest_path(tmp_path, XET_HASH).write_text(
            "../../outside\n/etc/passwd\nmodels--org--missing/blobs/etag\n"
        )

        assert sweep_shared_blob(store_entry, cache_dir=tmp_path) == len(CONTENT)
        assert not store_entry.exists()

    def test_manifest_io_error_leaks_instead_of_dropping_reference(self, tmp_path: Path) -> None:
        blob = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        manifest = shared_blob_manifest_path(tmp_path, XET_HASH)
        original_readlink = os.readlink

        def fail_blob_readlink(path):
            if Path(path) == blob:
                raise OSError("transient filesystem error")
            return original_readlink(path)

        with patch("huggingface_hub._shared_blobs.os.readlink", side_effect=fail_blob_readlink):
            assert sweep_shared_blob(store_entry, cache_dir=tmp_path) == 0

        assert manifest.read_text() == "models--org--repoA/blobs/etag\n"
        assert store_entry.exists()

    def test_link_and_gc_are_serialized_per_blob(self, tmp_path: Path) -> None:
        blob_a = _publish(tmp_path)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        blob_a.unlink()
        blob_b = tmp_path / "models--org--repoB" / "blobs" / "etag"
        blob_b.parent.mkdir(parents=True)
        manifest_flushed = threading.Event()
        finish_link = threading.Event()

        from huggingface_hub import _shared_blobs

        original_append = _shared_blobs._append_manifest_reference

        def slow_append(*args, **kwargs):
            original_append(*args, **kwargs)
            manifest_flushed.set()
            assert finish_link.wait(timeout=5)

        link_result: list[bool] = []
        gc_result: list[int] = []
        with patch("huggingface_hub._shared_blobs._append_manifest_reference", side_effect=slow_append):
            link_thread = threading.Thread(
                target=lambda: link_result.append(
                    try_link_from_shared_store(
                        blob_path=str(blob_b),
                        xet_hash=XET_HASH,
                        cache_dir=tmp_path,
                        expected_size=len(CONTENT),
                    )
                )
            )
            link_thread.start()
            assert manifest_flushed.wait(timeout=5)
            gc_thread = threading.Thread(
                target=lambda: gc_result.append(sweep_shared_blob(store_entry, cache_dir=tmp_path))
            )
            gc_thread.start()
            finish_link.set()
            link_thread.join(timeout=5)
            gc_thread.join(timeout=5)

        assert link_result == [True]
        assert gc_result == [0]
        assert blob_b.read_bytes() == CONTENT
        assert store_entry.exists()


@requires_reliable_symlinks
class TestScanAndDelete:
    def test_scan_keeps_repo_blob_as_first_hop(self, tmp_path: Path) -> None:
        commit = "a" * 40
        blob = _make_cached_file(tmp_path, "models--org--repo", commit, "file.bin", "11" * 32, ref="main")
        assert publish_blob_to_shared_store(
            blob_path=str(blob), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )

        report = scan_cache_dir(tmp_path)
        cached_file = next(iter(next(iter(report.repos)).revisions)).files
        assert next(iter(cached_file)).blob_path == blob
        assert report.warnings == []
        assert report.size_on_disk == len(CONTENT)

    def test_unmarked_blobs_directory_is_not_hidden(self, tmp_path: Path) -> None:
        _write_blob(shared_blobs_dir(tmp_path) / "user-file")

        report = scan_cache_dir(tmp_path)

        assert len(report.warnings) == 1

    def test_cache_total_counts_shared_payload_once_and_store_only_payloads(self, tmp_path: Path) -> None:
        rev_a, rev_b = "a" * 40, "b" * 40
        blob_a = _make_cached_file(tmp_path, "models--org--repoA", rev_a, "model.bin", "11" * 32, ref="main")
        assert publish_blob_to_shared_store(
            blob_path=str(blob_a), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        blob_b = _make_cached_file(tmp_path, "models--org--repoB", rev_b, "model.bin", "11" * 32, ref="main")
        blob_b.unlink()
        assert try_link_from_shared_store(
            blob_path=str(blob_b), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )

        report = scan_cache_dir(tmp_path)
        assert sum(repo.size_on_disk for repo in report.repos) == 2 * len(CONTENT)
        assert report.size_on_disk == len(CONTENT)

        shutil.rmtree(tmp_path / "models--org--repoA")
        shutil.rmtree(tmp_path / "models--org--repoB")
        assert scan_cache_dir(tmp_path).size_on_disk == len(CONTENT)

    def test_delete_uses_only_affected_manifests(self, tmp_path: Path) -> None:
        rev_a, rev_b = "a" * 40, "b" * 40
        blob_a = _make_cached_file(tmp_path, "models--org--repoA", rev_a, "model.bin", "11" * 32, ref="main")
        assert publish_blob_to_shared_store(
            blob_path=str(blob_a), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        blob_b = _make_cached_file(tmp_path, "models--org--repoB", rev_b, "model.bin", "11" * 32, ref="main")
        blob_b.unlink()
        assert try_link_from_shared_store(
            blob_path=str(blob_b), xet_hash=XET_HASH, cache_dir=tmp_path, expected_size=len(CONTENT)
        )
        unrelated = _publish(tmp_path, repo_folder="models--org--other", etag="other", xet_hash=OTHER_XET_HASH)

        strategy = scan_cache_dir(tmp_path).delete_revisions(rev_a)
        assert strategy.expected_freed_size == 0
        with patch("huggingface_hub._shared_blobs.sweep_shared_blob", wraps=sweep_shared_blob) as sweep:
            strategy.execute()
        sweep.assert_called_once_with(shared_blob_path(tmp_path, XET_HASH), cache_dir=tmp_path)
        assert blob_b.read_bytes() == CONTENT
        assert unrelated.read_bytes() == CONTENT

        strategy = scan_cache_dir(tmp_path).delete_revisions(rev_b)
        assert strategy.expected_freed_size == len(CONTENT)
        strategy.execute()
        assert not shared_blob_path(tmp_path, XET_HASH).exists()
        assert shared_blob_path(tmp_path, OTHER_XET_HASH).exists()


class TestDownloadIntegration:
    ETAG = "e7" * 32
    COMMIT = "d" * 40

    @pytest.fixture(autouse=True)
    def enable_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(constants, "HF_HUB_DISABLE_SHARED_BLOBS", False)

    def _download(self, cache_dir: Path, repo_id: str, xet_downloads: list[str], **kwargs) -> Path:
        metadata = (
            "https://example.com/file.bin",
            self.ETAG,
            self.COMMIT,
            len(CONTENT),
            XetFileData(file_hash=XET_HASH, refresh_route="https://example.com/refresh"),
            None,
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

    @requires_reliable_symlinks
    def test_second_repo_reuses_blob_without_download(self, tmp_path: Path) -> None:
        xet_downloads: list[str] = []

        path_a = self._download(tmp_path, "org/repoA", xet_downloads)
        path_b = self._download(tmp_path, "org/repoB", xet_downloads)

        store_entry = shared_blob_path(tmp_path, XET_HASH)
        assert path_a.read_bytes() == CONTENT
        assert path_b.read_bytes() == CONTENT
        assert path_a.resolve() == store_entry
        assert path_b.resolve() == store_entry
        assert store_entry.stat().st_nlink == 1
        assert xet_downloads == ["org/repoA"]

    @requires_reliable_symlinks
    def test_force_download_replaces_store_entry(self, tmp_path: Path) -> None:
        xet_downloads: list[str] = []
        path_a = self._download(tmp_path, "org/repoA", xet_downloads)
        path_b = self._download(tmp_path, "org/repoB", xet_downloads)
        old_inode = shared_blob_path(tmp_path, XET_HASH).stat().st_ino

        path_a = self._download(tmp_path, "org/repoA", xet_downloads, force_download=True)

        store_entry = shared_blob_path(tmp_path, XET_HASH)
        assert store_entry.stat().st_ino != old_inode
        assert path_a.resolve() == store_entry
        assert path_b.resolve() == store_entry
        assert xet_downloads == ["org/repoA", "org/repoA"]

    def test_symlink_failure_falls_back_to_regular_repo_blob(self, tmp_path: Path) -> None:
        xet_downloads: list[str] = []
        with patch("huggingface_hub._shared_blobs.os.symlink", side_effect=OSError("not supported")):
            path = self._download(tmp_path, "org/repoA", xet_downloads)

        repo_blob = tmp_path / "models--org--repoA" / "blobs" / self.ETAG
        assert path.read_bytes() == CONTENT
        assert not repo_blob.exists()  # regular no-symlink fallback moves it into the snapshot
        assert path.is_file() and not path.is_symlink()
        assert xet_downloads == ["org/repoA"]
        assert not shared_blob_path(tmp_path, XET_HASH).exists()

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
        assert not shared_blob_path(tmp_path, XET_HASH).exists()

    @requires_reliable_symlinks
    def test_disable_xet_bypasses_prepopulated_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        xet_downloads: list[str] = []
        self._download(tmp_path, "org/repoA", xet_downloads)
        store_entry = shared_blob_path(tmp_path, XET_HASH)
        monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
        http_content = CONTENT[::-1]
        metadata = (
            "https://example.com/file.bin",
            self.ETAG,
            self.COMMIT,
            len(http_content),
            XetFileData(file_hash=XET_HASH, refresh_route="https://example.com/refresh"),
            None,
        )

        def fake_http_get(url: str, temp_file, **kwargs) -> None:
            temp_file.write(http_content)

        with (
            patch("huggingface_hub.file_download._get_metadata_or_catch_error", return_value=metadata),
            patch("huggingface_hub.file_download.http_get", side_effect=fake_http_get),
            patch("huggingface_hub.file_download.is_xet_available", return_value=False),
        ):
            path = Path(hf_hub_download("org/repoB", "file.bin", cache_dir=str(tmp_path)))

        assert path.read_bytes() == http_content
        assert path.resolve() != store_entry

    @requires_reliable_symlinks
    def test_actual_snapshot_symlink_failure_preserves_repo_blob(self, tmp_path: Path) -> None:
        xet_downloads: list[str] = []
        with patch(
            "huggingface_hub.file_download.are_symlinks_supported",
            side_effect=[True, False],
        ):
            path = self._download(tmp_path, "org/repoA", xet_downloads)

        repo_blob = tmp_path / "models--org--repoA" / "blobs" / self.ETAG
        assert repo_blob.is_symlink()
        assert path.is_file() and not path.is_symlink()
        assert path.read_bytes() == CONTENT

    @requires_reliable_symlinks
    def test_store_enabled_by_default_and_can_be_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        xet_downloads: list[str] = []
        path = self._download(tmp_path / "enabled", "org/repoA", xet_downloads)
        assert path.resolve() == shared_blob_path(tmp_path / "enabled", XET_HASH)

        monkeypatch.setattr(constants, "HF_HUB_DISABLE_SHARED_BLOBS", True)
        path = self._download(tmp_path / "disabled", "org/repoA", xet_downloads)
        assert not shared_blobs_dir(tmp_path / "disabled").exists()
        assert "models--org--repoA/blobs" in path.resolve().as_posix()

    def test_degraded_no_symlink_mode_disables_store(self, tmp_path: Path) -> None:
        xet_downloads: list[str] = []
        with patch("huggingface_hub.file_download.are_symlinks_supported", return_value=False):
            path = self._download(tmp_path, "org/repoA", xet_downloads)
        assert path.read_bytes() == CONTENT
        assert not shared_blobs_dir(tmp_path).exists()


@requires_reliable_symlinks
def test_force_move_fallback_unlinks_symlink_without_mutating_target(tmp_path: Path) -> None:
    target = _write_blob(tmp_path / "target", b"old")
    destination = tmp_path / "destination"
    destination.symlink_to(target)
    source = _write_blob(tmp_path / "source", b"new")

    with patch("huggingface_hub.file_download.os.replace", side_effect=OSError("not supported")):
        _chmod_and_move(source, destination)

    assert target.read_bytes() == b"old"
    assert destination.read_bytes() == b"new"
    assert not destination.is_symlink()


@requires_reliable_symlinks
def test_force_move_fallback_restores_symlink_when_publication_fails(tmp_path: Path) -> None:
    target = _write_blob(tmp_path / "target", b"old")
    destination = tmp_path / "destination"
    destination.symlink_to(target)
    source = _write_blob(tmp_path / "source", b"new")
    original_move = shutil.move

    def fail_final_move(src, dst, **kwargs):
        if Path(dst) == destination:
            raise OSError("cannot publish staged file")
        return original_move(src, dst, **kwargs)

    with (
        patch("huggingface_hub.file_download.os.replace", side_effect=OSError("not supported")),
        patch("huggingface_hub.file_download.shutil.move", side_effect=fail_final_move),
        pytest.raises(OSError, match="cannot publish staged file"),
    ):
        _chmod_and_move(source, destination)

    assert target.read_bytes() == b"old"
    assert destination.is_symlink()
    assert destination.read_bytes() == b"old"
